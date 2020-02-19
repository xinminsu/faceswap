#!/usr/bin python3
""" Main entry point to the convert process of FaceSwap """

import logging
import re
import os
import sys
from threading import Event
from time import sleep

import cv2
import time
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from tqdm import tqdm

from scripts.fsmedia import Alignments, Images, PostProcess, finalize
from lib.serializer import get_serializer
from lib.convert import Converter
from lib.faces_detect import DetectedFace
from lib.gpu_stats import GPUStats
from lib.image import read_image_hash
from lib.multithreading import MultiThread, total_cpus
from lib.queue_manager import queue_manager
from lib.utils import FaceswapError, get_folder, get_image_paths
from plugins.extract.pipeline import Extractor, ExtractMedia
from plugins.plugin_loader import PluginLoader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Live():  # pylint:disable=too-few-public-methods
    """ The Faceswap Face Conversion Process.

    The conversion process is responsible for swapping the faces on source frames with the output
    from a trained model.

    It leverages a series of user selected post-processing plugins, executed from
    :class:`lib.convert.Converter`.

    The convert process is self contained and should not be referenced by any other scripts, so it
    contains no public properties.

    Parameters
    ----------
    arguments: :class:`argparse.Namespace`
        The arguments to be passed to the convert process as generated from Faceswap's command
        line arguments
    """
    def __init__(self, arguments):
        logger.debug("Initializing %s: (args: %s)", self.__class__.__name__, arguments)
        self._args = arguments

        self.batch = list()

        self._serializer = get_serializer("json")
        self._pre_process = PostProcess(arguments)
        self._writer = self._get_writer()
        self._extractor = self._load_extractor()

        self._batchsize = self._get_batchsize(self._queue_size)
        self._model = self._load_model()
        self._output_indices = {"face": self._model.largest_face_index,
                                "mask": self._model.largest_mask_index}

        self._predictor = self._model.converter(False)

        configfile = self._args.configfile if hasattr(self._args, "configfile") else None
        self._converter = Converter(self.output_size,
                                    self.coverage_ratio,
                                    self.draw_transparent,
                                    self.pre_encode,
                                    arguments,
                                    configfile=configfile)

        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def draw_transparent(self):
        """ bool: ``True`` if the selected writer's Draw_transparent configuration item is set
        otherwise ``False`` """
        return self._writer.config.get("draw_transparent", False)

    @property
    def pre_encode(self):
        """ python function: Selected writer's pre-encode function, if it has one,
        otherwise ``None`` """
        dummy = np.zeros((20, 20, 3), dtype="uint8")
        test = self._writer.pre_encode(dummy)
        retval = None if test is None else self._writer.pre_encode
        logger.debug("Writer pre_encode function: %s", retval)
        return retval

    @property
    def coverage_ratio(self):
        """ float: The coverage ratio that the model was trained at. """
        return self._model.training_opts["coverage_ratio"]

    @property
    def output_size(self):
        """ int: The size in pixels of the Faceswap model output. """
        return self._model.output_shape[0]

    @property
    def _queue_size(self):
        """ int: Size of the converter queues. 16 for single process otherwise 32 """
        if self._args.singleprocess:
            retval = 16
        else:
            retval = 32
        logger.debug(retval)
        return retval

    @property
    def _pool_processes(self):
        """ int: The number of threads to run in parallel. Based on user options and number of
        available processors. """
        retval = 1

        return retval

    @staticmethod
    def _get_batchsize(queue_size):
        """ Get the batch size for feeding the model.

        Sets the batch size to 1 if inference is being run on CPU, otherwise the minimum of the
        :attr:`self._queue_size` and 16.

        Returns
        -------
        int
            The batch size that the model is to be fed at.
        """
        logger.debug("Getting batchsize")
        is_cpu = GPUStats().device_count == 0
        batchsize = 1 if is_cpu else 16
        batchsize = min(queue_size, batchsize)
        logger.debug("Batchsize: %s", batchsize)
        logger.debug("Got batchsize: %s", batchsize)
        return batchsize

    def _add_queues(self):
        """ Add the queues for in, patch and out. """
        logger.debug("Adding queues. Queue size: %s", self._queue_size)
        for qname in ("convert_in", "convert_out", "patch"):
            queue_manager.add_queue(qname, self._queue_size)

    def process(self):
        """ The entry point for triggering the Conversion Process.

        Should only be called from  :class:`lib.cli.ScriptExecutor`
        """
        logger.debug("Starting Conversion")
        # queue_manager.debug_monitor(5)
        try:
            self._convert_images()

            logger.debug("Completed Conversion")
        except MemoryError as err:
            msg = ("Faceswap ran out of RAM running convert. Conversion is very system RAM "
                   "heavy, so this can happen in certain circumstances when you have a lot of "
                   "cpus but not enough RAM to support them all."
                   "\nYou should lower the number of processes in use by either setting the "
                   "'singleprocess' flag (-sp) or lowering the number of parallel jobs (-j).")
            raise FaceswapError(msg) from err

    def _convert_images(self):
        """ Start the multi-threaded patching process, monitor all threads for errors and join on
        completion. """
        logger.debug("Converting images")
        video_capture = cv2.VideoCapture(0)
        time.sleep(1)

        width = video_capture.get(3)  # float
        height = video_capture.get(4)  # float
        print("webcam dimensions = {} x {}".format(width, height))

        while True:
            ret, frame = video_capture.read()
            #frame = cv2.resize(frame, (640, 480))
            # flip image, because webcam inverts it and we trained the model the other way!
            frame = cv2.flip(frame, 1)
            image = self._convert_frame(frame, convert_colors=False)
            # flip it back
            #image = cv2.flip(image, 1)
            #image = cv2.resize(image, (640, 480))
            img = cv2.imread(self._writer.output_filename("result"))
            img = cv2.resize(img, (1280, 720))
            cv2.imshow('Video', img)
            # print("writing to screen")

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                video_capture.release()
                break

        cv2.destroyAllWindows()
        exit()

    def _convert_frame(self, frame, convert_colors=True):
        detected_faces = self._get_detected_faces("camera", frame)
        if len(detected_faces) == 0:
            return frame
        item = dict(filename="camera", image=frame, detected_faces=detected_faces)
        self.load_aligned(item)

        self.batch.clear()
        self.batch.append(item)
        detected_batch = [detected_face for item in self.batch
                          for detected_face in item["detected_faces"]]
        feed_faces = self._compile_feed_faces(detected_batch)
        predicted = self._predict(feed_faces, 1)

        pointer = 0
        for item in self.batch:
            num_faces = len(item["detected_faces"])
            if num_faces == 0:
                item["swapped_faces"] = np.array(list())
            else:
                item["swapped_faces"] = predicted[pointer:pointer + num_faces]
            pointer += num_faces

        imager = self._converter._patch_image(item)

        self._writer.write("result", imager)

        return imager

    def _convert_images2(self):
        """ Start the multi-threaded patching process, monitor all threads for errors and join on
        completion. """

        for filename, image in self._images.load():
            imager = self._process_image(filename, image)
            self._writer.write(filename, imager)



    def _process_image(self, filename, image):
        detected_faces = self._get_detected_faces(filename, image)
        item = dict(filename=filename, image=image, detected_faces=detected_faces)
        self._pre_process.do_actions(item)

        self.load_aligned(item)

        self.batch.clear()
        self.batch.append(item)
        detected_batch = [detected_face for item in self.batch
                          for detected_face in item["detected_faces"]]
        feed_faces = self._compile_feed_faces(detected_batch)
        predicted = self._predict(feed_faces, 1)

        pointer = 0
        for item in self.batch:
            num_faces = len(item["detected_faces"])
            if num_faces == 0:
                item["swapped_faces"] = np.array(list())
            else:
                item["swapped_faces"] = predicted[pointer:pointer + num_faces]
            pointer += num_faces

        imager = self._converter._patch_image(item)
        return imager

    def _get_writer(self):
        """ Load the selected writer plugin.

        Returns
        -------
        :mod:`plugins.convert.writer` plugin
            The requested writer plugin
        """
        args = [os.path.abspath(os.path.dirname(sys.argv[0]))]
        logger.debug("Writer args: %s", args)
        configfile = self._args.configfile if hasattr(self._args, "configfile") else None
        return PluginLoader.get_live("writer", self._args.writer)(*args,
                                                                       configfile=configfile)

    def _load_extractor(self):
        """ Load the CV2-DNN Face Extractor Chain.

        For On-The-Fly conversion we use a CPU based extractor to avoid stacking the GPU.
        Results are poor.

        Returns
        -------
        :class:`plugins.extract.Pipeline.Extractor`
            The face extraction chain to be used for on-the-fly conversion
        """

        logger.debug("Loading extractor")
        logger.warning("On-The-Fly conversion selected. This will use the inferior cv2-dnn for "
                       "extraction and will produce poor results.")
        logger.warning("It is recommended to generate an alignments file for your destination "
                       "video with Extract first for superior results.")
        extractor = Extractor(detector="cv2-dnn",
                              aligner="cv2-dnn",
                              masker="none",
                              multiprocess=True,
                              rotate_images=None,
                              min_size=20)
        extractor.launch()
        logger.debug("Loaded extractor")
        return extractor

    def _get_detected_faces(self, filename, image):
        """ Return the detected faces for the given image.

        If we have an alignments file, then the detected faces are created from that file. If
        we're running On-The-Fly then they will be extracted from the extractor.

        Parameters
        ----------
        filename: str
            The filename to return the detected faces for
        image: :class:`numpy.ndarray`
            The frame that the detected faces exist in

        Returns
        -------
        list
            List of :class:`lib.faces_detect.DetectedFace` objects
        """
        logger.trace("Getting faces for: '%s'", filename)
        if not self._extractor:
            detected_faces = self._alignments_faces(os.path.basename(filename), image)
        else:
            detected_faces = self._detect_faces(filename, image)
        logger.trace("Got %s faces for: '%s'", len(detected_faces), filename)
        return detected_faces

    def _detect_faces(self, filename, image):
        """ Extract the face from a frame for On-The-Fly conversion.

        Pulls detected faces out of the Extraction pipeline.

        Parameters
        ----------
        filename: str
            The filename to return the detected faces for
        image: :class:`numpy.ndarray`
            The frame that the detected faces exist in

        Returns
        -------
        list
            List of :class:`lib.faces_detect.DetectedFace` objects
         """
        self._extractor.input_queue.put(ExtractMedia(filename, image))
        faces = next(self._extractor.detected_faces())

        final_faces = [face for face in faces.detected_faces]
        return final_faces

    def _load_model(self):
        """ Load the Faceswap model.

        Returns
        -------
        :mod:`plugins.train.model` plugin
            The trained model in the specified model folder
        """
        logger.debug("Loading Model")
        model_dir = get_folder(self._args.model_dir, make_folder=False)
        if not model_dir:
            raise FaceswapError("{} does not exist.".format(self._args.model_dir))
        trainer = self._get_model_name(model_dir)
        gpus = 1 if not hasattr(self._args, "gpus") else self._args.gpus
        model = PluginLoader.get_model(trainer)(model_dir, gpus, predict=True)
        logger.debug("Loaded Model")
        return model

    def _get_model_name(self, model_dir):
        """ Return the name of the Faceswap model used.

        If a "trainer" option has been selected in the command line arguments, use that value,
        otherwise retrieve the name of the model from the model's state file.

        Parameters
        ----------
        model_dir: str
            The folder that contains the trained Faceswap model

        Returns
        -------
        str
            The name of the Faceswap model being used.

        """
        if hasattr(self._args, "trainer") and self._args.trainer:
            logger.debug("Trainer name provided: '%s'", self._args.trainer)
            return self._args.trainer

        statefile = [fname for fname in os.listdir(str(model_dir))
                     if fname.endswith("_state.json")]
        if len(statefile) != 1:
            raise FaceswapError("There should be 1 state file in your model folder. {} were "
                                "found. Specify a trainer with the '-t', '--trainer' "
                                "option.".format(len(statefile)))
        statefile = os.path.join(str(model_dir), statefile[0])

        state = self._serializer.load(statefile)
        trainer = state.get("name", None)

        if not trainer:
            raise FaceswapError("Trainer name could not be read from state file. "
                                "Specify a trainer with the '-t', '--trainer' option.")
        logger.debug("Trainer from state file: '%s'", trainer)
        return trainer

    def _alignments_faces(self, frame_name, image):
        """ Return detected faces from an alignments file.

        Parameters
        ----------
        frame_name: str
            The name of the frame to return the detected faces for
        image: :class:`numpy.ndarray`
            The frame that the detected faces exist in

        Returns
        -------
        list
            List of :class:`lib.faces_detect.DetectedFace` objects
        """
        if not self._check_alignments(frame_name):
            return list()

        faces = self._alignments.get_faces_in_frame(frame_name)
        detected_faces = list()

        for rawface in faces:
            face = DetectedFace()
            face.from_alignment(rawface, image=image)
            detected_faces.append(face)
        return detected_faces

    def _check_alignments(self, frame_name):
        """ Ensure that we have alignments for the current frame.

        If we have no alignments for this image, skip it and output a message.

        Parameters
        ----------
        frame_name: str
            The name of the frame to check that we have alignments for

        Returns
        -------
        bool
            ``True`` if we have alignments for this face, otherwise ``False``
        """
        have_alignments = self._alignments.frame_exists(frame_name)
        if not have_alignments:
            tqdm.write("No alignment found for {}, "
                       "skipping".format(frame_name))
        return have_alignments

    @property
    def _input_size(self):
        """ int: The size in pixels of the Faceswap model input. """
        return self._model.input_shape[0]

    @property
    def coverage_ratio(self):
        """ float: The coverage ratio that the model was trained at. """
        return self._model.training_opts["coverage_ratio"]

    @property
    def output_size(self):
        """ int: The size in pixels of the Faceswap model output. """
        return self._model.output_shape[0]

    def load_aligned(self, item):
        """ Load the model's feed faces and the reference output faces.

        For each detected face in the incoming item, load the feed face and reference face
        images, correctly sized for input and output respectively.

        Parameters
        ----------
        item: dict
            The incoming image and list of :class:`~lib.faces_detect.DetectedFace` objects

        """
        logger.trace("Loading aligned faces: '%s'", item["filename"])
        for detected_face in item["detected_faces"]:
            detected_face.load_feed_face(item["image"],
                                         size=self._input_size,
                                         coverage_ratio=self.coverage_ratio,
                                         dtype="float32")
            if self._input_size == self.output_size:
                detected_face.reference = detected_face.feed
            else:
                detected_face.load_reference_face(item["image"],
                                                  size=self.output_size,
                                                  coverage_ratio=self.coverage_ratio,
                                                  dtype="float32")
        logger.trace("Loaded aligned faces: '%s'", item["filename"])

    @staticmethod
    def _compile_feed_faces(detected_faces):
        """ Compile a batch of faces for feeding into the Predictor.

        Parameters
        ----------
        detected_faces: list
            List of `~lib.faces_detect.DetectedFace` objects

        Returns
        -------
        :class:`numpy.ndarray`
            A batch of faces ready for feeding into the Faceswap model.
        """
        logger.trace("Compiling feed face. Batchsize: %s", len(detected_faces))
        feed_faces = np.stack([detected_face.feed_face[..., :3]
                               for detected_face in detected_faces]) / 255.0
        logger.trace("Compiled Feed faces. Shape: %s", feed_faces.shape)
        return feed_faces

    def _predict(self, feed_faces, batch_size=None):
        """ Run the Faceswap models' prediction function.

        Parameters
        ----------
        feed_faces: :class:`numpy.ndarray`
            The batch to be fed into the model
        batch_size: int, optional
            Used for plaidml only. Indicates to the model what batch size is being processed.
            Default: ``None``

        Returns
        -------
        :class:`numpy.ndarray`
            The swapped faces for the given batch
        """
        logger.trace("Predicting: Batchsize: %s", len(feed_faces))
        feed = [feed_faces]
        if self._model.feed_mask:
            feed.append(np.repeat(self._input_mask, feed_faces.shape[0], axis=0))
        logger.trace("Input shape(s): %s", [item.shape for item in feed])

        predicted = self._predictor(feed, batch_size=batch_size)
        predicted = predicted if isinstance(predicted, list) else [predicted]
        logger.trace("Output shape(s): %s", [predict.shape for predict in predicted])

        predicted = self._filter_multi_out(predicted)

        # Compile masks into alpha channel or keep raw faces
        predicted = np.concatenate(predicted, axis=-1) if len(predicted) == 2 else predicted[0]
        predicted = predicted.astype("float32")

        logger.trace("Final shape: %s", predicted.shape)
        return predicted

    def _filter_multi_out(self, predicted):
        """ Filter the model output to just the required image.

        Some models have multi-scale outputs, so just make sure we take the largest
        output.

        Parameters
        ----------
        predicted: :class:`numpy.ndarray`
            The predictions retrieved from the Faceswap model.

        Returns
        -------
        :class:`numpy.ndarray`
            The predictions with any superfluous outputs removed.
        """
        if not predicted:
            return predicted
        face = predicted[self._output_indices["face"]]
        mask_idx = self._output_indices["mask"]
        mask = predicted[mask_idx] if mask_idx is not None else None
        predicted = [face, mask] if mask is not None else [face]
        logger.trace("Filtered output shape(s): %s", [predict.shape for predict in predicted])
        return predicted




