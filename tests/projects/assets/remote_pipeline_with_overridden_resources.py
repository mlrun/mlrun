import kfp.dsl
import kubernetes.client

import mlrun

overridden_affinity = kubernetes.client.V1Affinity(
    node_affinity=kubernetes.client.V1NodeAffinity(
        required_during_scheduling_ignored_during_execution=kubernetes.client.V1NodeSelector(
            node_selector_terms=[
                kubernetes.client.V1NodeSelectorTerm(
                    match_expressions=kubernetes.client.V1NodeSelectorRequirement(
                        key="override_key", operator="NoSchedule", values=["haha"]
                    )
                )
            ]
        )
    )
)


def func1(context, p1=1):
    context.log_result("accuracy", p1 * 2)


@kfp.dsl.pipeline(name="remote_pipeline", description="tests remote pipeline")
def pipeline():
    run1 = mlrun.run_function("func1", handler="func1", params={"p1": 9})
    run1.container.resources.limits = {"cpu": "2000m", "memory": "4G"}
    run2 = mlrun.run_function("func2", handler="func1", params={"p1": 9})
    run2.affinity = overridden_affinity
