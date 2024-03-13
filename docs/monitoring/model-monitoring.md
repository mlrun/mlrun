(monitoring)=

# Model monitoring
```{note}
This is currently a beta feature. It currently does not cover batch monitoring.
```

Adding the ability for a user to define an monitoring app that should be run on a set of model end-points. 

Within this phase there are a few restrictions that will be addressed on subsequent phase:  

All apps will be running on all model endpoints 

Scheduling is based on the job schedules. There is no "monitoring policy" yet

There is a single job - dividing to several jobs by different sched/app/others will be done on the next phase. 

Drift type will remain as is - drift is detected when it is identified by the threshold