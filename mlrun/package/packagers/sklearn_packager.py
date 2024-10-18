import os
from typing import Tuple, Type

import sklearn
from sklearn.pipeline import Pipeline

from mlrun import ArtifactType
from mlrun.artifacts import Artifact, ModelArtifact, get_model
from mlrun.datastore import DataItem
from mlrun.frameworks.auto_mlrun import AutoMLRun
from mlrun.package import DefaultPackager
from mlrun.package.utils import Pickler, TypeHintUtils


class SklearnModelPack(DefaultPackager):
    """
    Sklearn packager for models that inherit from sklearn.base.BaseEstimator
    """

    PACKABLE_OBJECT_TYPE = sklearn.base.BaseEstimator
    # PACKABLE_OBJECT_TYPE = SKLearnModelHandler
    DEFAULT_PACKING_ARTIFACT_TYPE = ArtifactType.MODEL
    DEFAULT_UNPACKING_ARTIFACT_TYPE = ArtifactType.MODEL
    PACK_SUBCLASSES = True
    DEFAULT_PICKLE_MODULE = "cloudpickle"

    @classmethod
    def pack_model(
        cls,
        obj: str,
        key: str,
    ) -> Tuple[Artifact, dict]:
        """
        Pack a sklearn model/pipeline (any subclass of sklearn.base.BaseEstimator)

        :param obj:            The sklearn object to pack.
        :param key:            The key to use for the artifact.

        :return: The packed artifact and instructions.
        """

        # Uses AutoMLRun to get the SKLearnMLRunInterface
        mlrun_model_interface = AutoMLRun.get_interface(obj)

        # Remove interface if model is wrapped
        if mlrun_model_interface.is_applied(obj):
            mlrun_model_interface.remove_interface(obj)

        # If model is a Pipeline object, each step in it can be wrapped by interface
        if isinstance(obj, Pipeline):
            for step in obj.steps:

                # Objects in Pipeline can be of different frameworks, so we get the relevant interface
                step_mlrun_model_interface = AutoMLRun.get_interface(step[1])
                if step_mlrun_model_interface.is_applied(step[1]):
                    step_mlrun_model_interface.remove_interface(step[1])

        art_obj, art_dict = cls.pack_object(obj=obj, key=key)

        artifact = ModelArtifact(
            key=art_obj.key,
            model_dir=os.path.dirname(art_obj.src_path),
            model_file=os.path.basename(art_obj.src_path),
        )

        return artifact, art_dict

    @classmethod
    def unpack_model(
        cls,
        data_item: DataItem,
        pickle_module_name: str = DEFAULT_PICKLE_MODULE,
        object_module_name: str = None,
        python_version: str = None,
        pickle_module_version: str = None,
        object_module_version: str = None,
    ) -> str:
        """
        Unpack the data item's object, unpickle it using the instructions, and return.

        :Warnings of mismatching python and module versions between the original pickling interpreter and this one may
        be raised.

        :param data_item:             The data item holding the pkl file.
        :param pickle_module_name:    Module to use for unpickling the object.
        :param object_module_name:    The original object's module. Used to verify that the current interpreter object
                                      module version matches the pickled object version before unpickling the object.
        :param python_version:        The python version in which the original object was pickled. Used to verify that
                                      the current interpreter python version matches the pickled object version before
                                      unpickling the object.
        :param pickle_module_version: The pickle module version. Used to verify that the current interpreter module
                                      version matches the one that pickled the object before unpickling it.
        :param object_module_version: The original object's module version to match to the interpreter's module version.

        :return: The un-pickled python object.
        """

        # First we get the model file path
        model_file, _, _ = get_model(model_dir=data_item)

        # Add the pickle path to the clearing list:
        cls.add_future_clearing_path(path=model_file)

        # Finally we return the unpickled model
        return Pickler.unpickle(
            pickle_path=model_file,
            pickle_module_name=pickle_module_name,
            object_module_name=object_module_name,
            python_version=python_version,
            pickle_module_version=pickle_module_version,
            object_module_version=object_module_version,
        )

    @classmethod
    def is_unpackable(
        cls, data_item: DataItem, type_hint: Type, artifact_type: str = None
    ) -> bool:
        """
        Check if this packager can unpack an input according to the user given type hint and the provided artifact type.

        The default implementation tries to match the packable object type of this packager to the given type hint, if
        it does match, it will look for the artifact type in the list returned from `get_supported_artifact_types`.

        :param data_item:     The input data item to check if unpackable.
        :param type_hint:     The type hint of the input to unpack.
        :param artifact_type: The artifact type to unpack the object as.

        :return: True if unpackable and False otherwise.
        """
        # Check type (ellipses means any type):
        if cls.PACKABLE_OBJECT_TYPE is not ...:
            if not TypeHintUtils.is_matching(
                object_type=type_hint,
                type_hint=cls.PACKABLE_OBJECT_TYPE,
                reduce_type_hint=False,
            ):
                return False

        # Check the artifact type:
        if artifact_type and artifact_type not in cls.get_supported_artifact_types():
            return False

        # Unpackable:
        return True
