def transform(event):
    if event.get("Product") == "Mouse":
        event["Product"] = "Mickey Mouse"
    return event
