import mlrun
from kfp import dsl

    
@dsl.pipeline(
    name="GenAI demo"
)

def kfpipeline(data_set, cache_dir):
    
    project = mlrun.get_current_project()
    
    fetch = project.run_function(
        function="fetch-vectordb-data",
        name="fetch-vectordb-data-run",
        handler="handler",
        params = {"data_set" : data_set},
        outputs=['vector-db-dataset']
    )
    
    
    vectordb_build = project.run_function(
        function="build-vectordb",
        inputs={"df" : fetch.outputs["vector-db-dataset"]},
        params={"cache_dir" : cache_dir},
        handler="handler_chroma"
    )

    deploy = project.deploy_function("serve-llm").after(vectordb_build)
