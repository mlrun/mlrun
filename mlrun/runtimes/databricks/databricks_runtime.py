from base64 import b64decode, b64encode
from mlrun.runtimes.kubejob import KubejobRuntime
from mlrun.model import RunObject

class DatabricksRuntime(KubejobRuntime):
    kind = "databricks"
    _is_remote = True

    def get_internal_code(self, runobj: RunObject):
        """
        Return the internal function code.
        """
        encoded_code = (
            self.spec.build.functionSourceCode if hasattr(self.spec, "build") else None
        )
        decoded_code = b64decode(encoded_code).decode("utf-8")
        code = _databricks_script_code + decoded_code
        if runobj.spec.handler:
            code += f"\n{runobj.spec.handler}(**handler_arguments)\n"
        code = b64encode(code.encode("utf-8")).decode("utf-8")
        return code

    def _pre_run(self, runspec: RunObject, execution):
        internal_code = self.get_internal_code(runspec)
        if internal_code:
            runspec.spec.parameters["mlrun_internal_code"] = self.get_internal_code(
                runspec
            )

            current_file = os.path.abspath(__file__)
            current_dir = os.path.dirname(current_file)
            databricks_runtime_wrap_path = os.path.join(
                current_dir, "databricks/databricks_runtime_wrapper.py"
            )
            with open(
                databricks_runtime_wrap_path, "r"
            ) as databricks_runtime_wrap_file:
                wrap_code = databricks_runtime_wrap_file.read()
                wrap_code = b64encode(wrap_code.encode("utf-8")).decode("utf-8")
            self.spec.build.functionSourceCode = wrap_code
            runspec.spec.handler = "run_mlrun_databricks_job"
        else:
            raise ValueError("Databricks function must be provided with user code")


_databricks_script_code = """

import argparse
import json
parser = argparse.ArgumentParser()
parser.add_argument('handler_arguments')
handler_arguments = parser.parse_args().handler_arguments
handler_arguments = json.loads(handler_arguments)

"""
