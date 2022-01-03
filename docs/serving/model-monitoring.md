# Model Monitoring

Model activities can be tracked into a real-time stream and time-series DB, the monitoring data
used to create real-time dashboards and track model accuracy and drift. 
to set the streaming option specify the following function spec attributes:

to add tracking to your model add tracking parameters to your function:

    fn.set_tracking(stream_path, batch, sample)

* **stream_path** - the v3io stream path (e.g. `v3io:///users/..`)
* **sample** -  optional, sample every N requests
* **batch** -  optional, send micro-batches every N requests


