from kfp import dsl

funcs = {}


@dsl.pipeline(name="Example pipeline", description="some pipeline description.")
def kfpipeline():

    # analyze our dataset
    funcs["describe"].as_step(
        name="summary", params={"label_column": "labels"},
    )

    # train with hyper-paremeters
    funcs["trainer-function"].as_step(name="trainer-function",)
