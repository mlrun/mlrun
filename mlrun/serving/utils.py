from mlrun.utils import get_in, update_in


def _extract_input_data(input_path, body):
    if input_path:
        if not hasattr(body, "__getitem__"):
            raise TypeError("input_path parameter supports only dict-like event bodies")
        return get_in(body, input_path)
    return body


def _update_result_body(result_path, event_body, result):
    if result_path and event_body:
        if not hasattr(event_body, "__getitem__"):
            raise TypeError(
                "result_path parameter supports only dict-like event bodies"
            )
        update_in(event_body, result_path, result)
    else:
        event_body = result
    return event_body
