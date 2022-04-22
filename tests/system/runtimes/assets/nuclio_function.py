def handler(context, event):
    context.logger.info("Hello world")

    return context.Response(
        body="Hello, from nuclio :]",
        headers={},
        content_type="text/plain",
        status_code=200,
    )
