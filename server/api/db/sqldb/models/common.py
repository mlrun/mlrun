_tagged = None
_labeled = None
_with_notifications = None
_classes = None
_table2cls = None


def post_table_definitions(base_cls):
    global _tagged
    global _labeled
    global _with_notifications
    global _classes
    global _table2cls
    _tagged = [cls for cls in base_cls.__subclasses__() if hasattr(cls, "Tag")]
    _labeled = [cls for cls in base_cls.__subclasses__() if hasattr(cls, "Label")]
    _with_notifications = [
        cls for cls in base_cls.__subclasses__() if hasattr(cls, "Notification")
    ]
    _classes = [cls for cls in base_cls.__subclasses__()]
    _table2cls = {cls.__table__.name: cls for cls in base_cls.__subclasses__()}
