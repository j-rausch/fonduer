import logging
from multiprocessing import JoinableQueue, Process
from queue import Empty

from fonduer.meta import Meta, new_sessionmaker

try:
    from IPython import get_ipython

    if "IPKernelApp" not in get_ipython().config:
        raise ImportError("console")
except (AttributeError, ImportError):
    from tqdm import tqdm
else:
    from tqdm import tqdm_notebook as tqdm


QUEUE_TIMEOUT = 3

# Grab pointer to global metadata
_meta = Meta.init()


class UDFRunner(object):
    """
    Class to run UDFs in parallel using simple queue-based multiprocessing
    setup.
    """

    def __init__(self, udf_class, **udf_init_kwargs):
        self.logger = logging.getLogger(__name__)
        self.udf_class = udf_class
        self.sql_session_class = SessionProcess
        self.sql_sessions = []
        self.udf_init_kwargs = udf_init_kwargs
        self.udfs = []
        self.pb = None

    def apply(
        self, xs, clear=True, parallelism=None, progress_bar=True, count=None, **kwargs
    ):
        """
        Apply the given UDF to the set of objects xs, either single or
        multi-threaded, and optionally calling clear() first.
        """
        # Clear everything downstream of this UDF if requested
        if clear:
            self.logger.info("Clearing existing...")
            Session = new_sessionmaker()
            session = Session()
            self.clear(session, **kwargs)
            session.commit()
            session.close()

        # Execute the UDF
        self.logger.info("Running UDF...")

        # Setup progress bar
        if progress_bar and hasattr(xs, "__len__") or count is not None:
            self.logger.debug("Setting up progress bar...")
            n = count if count is not None else len(xs)
            self.pb = tqdm(total=n)

        if parallelism is None or parallelism < 2:
            self.apply_st(xs, clear=clear, count=count, **kwargs)
        else:
            self.apply_mt(xs, parallelism, clear=clear, **kwargs)

        # Close progress bar
        if self.pb is not None:
            self.logger.debug("Closing progress bar...")
            self.pb.close()

    def clear(self, session, **kwargs):
        raise NotImplementedError()

    def apply_st(self, xs, count, **kwargs):
        """Run the UDF single-threaded, optionally with progress bar"""
        udf = self.udf_class(**self.udf_init_kwargs)
        sql_session_process = self.sql_session_class()

        # Run single-thread
        for x in xs:
            if self.pb is not None:
                self.pb.update(1)
            all_sentences = [y for y in udf.apply(x, **kwargs)]
            sql_session_process.session.add_all(all_sentences)
            # udf.session.add_all(y for y in udf.apply(x, **kwargs))

        # Commit session and close progress bar if applicable
        # udf.session.commit()
        sql_session_process.session.commit()

    def apply_mt(self, xs, parallelism, **kwargs):
        """Run the UDF multi-threaded using python multiprocessing"""
        if not _meta.postgres:
            raise ValueError("Fonduer must use PostgreSQL as a database backend.")

        # Fill a JoinableQueue with input objects
        in_queue = JoinableQueue()
        for x in xs:
            in_queue.put(x)

        # Use an output queue to track multiprocess progress
        out_queue = JoinableQueue()

        in_queue_sentences = JoinableQueue()
        out_queue_sentences = JoinableQueue()

        # Track progress counts
        total_count = in_queue.qsize()
        count = 0

        # Start UDF and SQL session processes
        for i in range(parallelism):
            udf = self.udf_class(
                in_queue=in_queue,
                out_queue=out_queue,
                worker_id=i,
                **self.udf_init_kwargs
            )
            udf.apply_kwargs = kwargs
            self.udfs.append(udf)

            sql_session = self.sql_session_class(
                in_queue_sentences=in_queue_sentences,
                out_queue_sentences=out_queue_sentences,
                worker_id=i,
            )
            self.sql_sessions.append(sql_session)

        # Start the UDF processes, and then join on their completion
        for udf in self.udfs:
            udf.start()

        # Start sql sessions
        for sql_session in self.sql_sessions:
            sql_session.start()

        while any([udf.is_alive() for udf in self.udfs]) and count < total_count:
            doc_sentences = out_queue.get()
            in_queue_sentences.put(doc_sentences)

            y = out_queue_sentences.get()
            # Update progress bar whenever an item is processed
            if y == SessionProcess.TASK_DONE:
                count += 1
                if self.pb is not None:
                    self.pb.update(1)
            else:
                raise ValueError("Got non-sentinal output.")

        # Tell sql sessions to stop waiting for new queue items
        in_queue_sentences.put(SessionProcess.QUEUE_CLOSED)

        for udf in self.udfs:
            udf.join()

        if (
            any([sql_session.is_alive() for sql_session in self.sql_sessions])
            and count < total_count
        ):
            y = out_queue_sentences.get()
            # Update progress bar whenever an item is processed
            if y == SessionProcess.TASK_DONE:
                count += 1
                if self.pb is not None:
                    self.pb.update(1)
            else:
                raise ValueError("Got non-sentinal output.")

        for sql_session in self.sql_sessions:
            sql_session.join()

        # Terminate and flush the processes
        for udf in self.udfs:
            udf.terminate()

        for sql_session in self.sql_sessions:
            sql_session.terminate()

        self.udfs = []
        self.sql_sessions = []


class SessionProcess(Process):
    TASK_DONE = "done"
    QUEUE_CLOSED = "queueclosed"

    def __init__(self, in_queue_sentences=None, out_queue_sentences=None, worker_id=0):
        """
        in_queue_sentences: A Queue of input objects to process;
        primarily for running in parallel
        """
        Process.__init__(self)
        self.daemon = True
        self.in_queue_sentences = in_queue_sentences
        self.out_queue_sentences = out_queue_sentences
        self.worker_id = worker_id

        # Each UDF starts its own Engine
        # See SQLalchemy, using connection pools with multiprocessing.
        Session = new_sessionmaker()
        self.session = Session()

    def run(self):
        """
        This method is called when the UDF is run as a Process in a
        multiprocess setting The basic routine is: get from JoinableQueue,
        apply, put / add outputs, loop
        """
        while True:
            try:
                x = self.in_queue_sentences.get(True, QUEUE_TIMEOUT)
                if x == SessionProcess.QUEUE_CLOSED:
                    self.in_queue_sentences.put(SessionProcess.QUEUE_CLOSED)
                    break
                self.session.add_all(x)
                self.session.commit()
                self.in_queue_sentences.task_done()
                self.out_queue_sentences.put(SessionProcess.TASK_DONE)
            except Empty:
                continue  # keep waiting for new elements to be added to queue
        self.session.close()

    def apply(self, x, **kwargs):
        """This function takes in an object, and returns a generator / set / list"""
        raise NotImplementedError()


class UDF(Process):
    # TASK_DONE = "done"

    def __init__(self, in_queue=None, out_queue=None, worker_id=0):
        """
        in_queue: A Queue of input objects to process; primarily for running in parallel
        """
        Process.__init__(self)
        self.daemon = True
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.worker_id = worker_id

        #        # Each UDF starts its own Engine
        #        # See SQLalchemy, using connection pools with multiprocessing.
        #        Session = new_sessionmaker()
        #        self.session = Session()

        # We use a workaround to pass in the apply kwargs
        self.apply_kwargs = {}

    def run(self):
        """
        This method is called when the UDF is run as a Process in a
        multiprocess setting The basic routine is: get from JoinableQueue,
        apply, put / add outputs, loop
        """
        while True:
            try:
                x = self.in_queue.get(True, QUEUE_TIMEOUT)
                # self.session.add_all(y for y in self.apply(x, **self.apply_kwargs))
                all_sentences = [y for y in self.apply(x, **self.apply_kwargs)]
                self.in_queue.task_done()
                self.out_queue.put(all_sentences)
            except Empty:
                break

    #        self.session.commit()
    #        self.session.close()

    def apply(self, x, **kwargs):
        """This function takes in an object, and returns a generator / set / list"""
        raise NotImplementedError()
