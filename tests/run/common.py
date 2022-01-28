import pandas as pd


def my_func(context, p1=1, p2="a-string", input_name="infile.txt"):
    print(f"Run: {context.name} (uid={context.uid})")
    print(f"Params: p1={p1}, p2={p2}\n")
    input_str = context.get_input(input_name).get()
    print(f"file\n{input_str}\n")

    context.log_result("accuracy", p1 * 2)
    context.logger.info("some info")
    context.logger.debug("debug info")
    context.log_metric("loss", 7)
    context.log_artifact("chart", body="abc")

    raw_data = {
        "first_name": ["Jason", "Molly", "Tina", "Jake", "Amy"],
        "last_name": ["Miller", "Jacobson", "Ali", "Milner", "Cooze"],
        "age": [42, 52, 36, 24, 73],
        "postTestScore": [25, 94, 57, 62, 70],
    }
    df = pd.DataFrame(
        raw_data, columns=["first_name", "last_name", "age", "postTestScore"]
    )
    context.log_dataset("mydf", df=df)
