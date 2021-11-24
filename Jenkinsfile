@Library('pipelinex@development') _
import com.iguazio.pipelinex.DockerRepo

workDir = '/home/jenkins'
podLabel = 'mlrun-release'
gitProject = 'mlrun'
gitProjectUser = 'mlrun'
gitProjectUI = 'ui'
dockerTag = env.TAG_NAME.replaceFirst(/^v/, '')

podTemplate(
    label: podLabel,
    containers: [
        containerTemplate(name: 'jnlp', image: 'jenkins/jnlp-slave:4.0.1-1', workingDir: workDir, resourceRequestCpu: '2000m', resourceLimitCpu: '2000m', resourceRequestMemory: '2048Mi', resourceLimitMemory: '2048Mi'),
        containerTemplate(name: 'base-build', image: 'iguazioci/alpine-base-build:ae7e534841e68675d15f4bd98f07197aed5591af', workingDir: workDir, ttyEnabled: true, command: 'cat'),
        containerTemplate(name: 'python37', image: 'python:3.7-stretch', workingDir: workDir, ttyEnabled: true, command: 'cat'),
    ],
    volumes: [
        hostPathVolume(mountPath: '/var/run/docker.sock', hostPath: '/var/run/docker.sock'),
    ],
) {
    node(podLabel) {
        common.notify_slack {
            withCredentials([
                string(credentialsId: 'iguazio-prod-git-user-token', variable: 'GIT_TOKEN')
            ]) {

                container('base-build') {
                    stage("git clone") {
                        checkout scm
                    }
                    stage("build ${gitProject}/api in dood") {
                        println(common.shellc("MLRUN_VERSION=${dockerTag} make api"))
                    }
                    dockerx.images_push_multi_registries(["${gitProject}/mlrun-api:${dockerTag}"], [DockerRepo.ARTIFACTORY_IGUAZIO, DockerRepo.MLRUN_DOCKER_HUB, DockerRepo.MLRUN_QUAY_IO])

                    stage("build ${gitProject}/mlrun in dood") {
                        println(common.shellc("MLRUN_VERSION=${dockerTag} make mlrun"))
                    }
                    dockerx.images_push_multi_registries(["${gitProject}/mlrun:${dockerTag}"], [DockerRepo.ARTIFACTORY_IGUAZIO, DockerRepo.MLRUN_DOCKER_HUB, DockerRepo.MLRUN_QUAY_IO])

                    stage("build ${gitProject}/jupyter in dood") {
                        println(common.shellc("MLRUN_VERSION=${dockerTag} make jupyter"))
                    }
                    dockerx.images_push_multi_registries(["${gitProject}/jupyter:${dockerTag}"], [DockerRepo.ARTIFACTORY_IGUAZIO, DockerRepo.MLRUN_DOCKER_HUB, DockerRepo.MLRUN_QUAY_IO])

                    stage("build ${gitProject}/base in dood") {
                        println(common.shellc("MLRUN_VERSION=${dockerTag} make base"))
                    }
                    dockerx.images_push_multi_registries(["${gitProject}/ml-base:${dockerTag}"], [DockerRepo.ARTIFACTORY_IGUAZIO, DockerRepo.MLRUN_DOCKER_HUB, DockerRepo.MLRUN_QUAY_IO])

                    stage("build ${gitProject}/base-legacy in dood") {
                        println(common.shellc("MLRUN_VERSION=${dockerTag} make base-legacy"))
                    }
                    dockerx.images_push_multi_registries(["${gitProject}/ml-base:${dockerTag}-py36"], [DockerRepo.ARTIFACTORY_IGUAZIO, DockerRepo.MLRUN_DOCKER_HUB, DockerRepo.MLRUN_QUAY_IO])

                    stage("build ${gitProject}/models in dood") {
                        println(common.shellc("MLRUN_VERSION=${dockerTag} make models"))
                    }
                    dockerx.images_push_multi_registries(["${gitProject}/ml-models:${dockerTag}"], [DockerRepo.ARTIFACTORY_IGUAZIO, DockerRepo.MLRUN_DOCKER_HUB, DockerRepo.MLRUN_QUAY_IO])

                    stage("build ${gitProject}/models-legacy in dood") {
                        println(common.shellc("MLRUN_VERSION=${dockerTag} make models-legacy"))
                    }
                    dockerx.images_push_multi_registries(["${gitProject}/ml-models:${dockerTag}-py36"], [DockerRepo.ARTIFACTORY_IGUAZIO, DockerRepo.MLRUN_DOCKER_HUB, DockerRepo.MLRUN_QUAY_IO])

                    stage("build ${gitProject}/models-gpu in dood") {
                        println(common.shellc("MLRUN_VERSION=${dockerTag} make models-gpu"))
                    }
                    dockerx.images_push_multi_registries(["${gitProject}/ml-models-gpu:${dockerTag}"], [DockerRepo.ARTIFACTORY_IGUAZIO, DockerRepo.MLRUN_DOCKER_HUB, DockerRepo.MLRUN_QUAY_IO])

                    stage("build ${gitProject}/models-gpu-legacy in dood") {
                        println(common.shellc("MLRUN_VERSION=${dockerTag} make models-gpu-legacy"))
                    }
                    dockerx.images_push_multi_registries(["${gitProject}/ml-models-gpu:${dockerTag}-py36"], [DockerRepo.ARTIFACTORY_IGUAZIO, DockerRepo.MLRUN_DOCKER_HUB, DockerRepo.MLRUN_QUAY_IO])
                }

                container('jnlp') {
                    common.conditional_stage('Create mlrun/ui release', "${env.TAG_NAME}" != "unstable") {
                        def mlrun_github_client = new Githubc(gitProjectUser, gitProject, GIT_TOKEN, env.TAG_NAME, this)
                        def ui_github_client = new Githubc(gitProjectUser, gitProjectUI, GIT_TOKEN, this)
                        def source_branch = mlrun_github_client.getReleasecommittish()

                        println("Source branch is: ${source_branch}, using this as source for ${gitProject}/${gitProjectUI}")
                        println("You are responsible to make sure that this branch exists in ${gitProject}/${gitProjectUI}!")

                        if (!source_branch) {
                            error("Could not get source branch from tag")
                        }

                        ui_github_client.createRelease(source_branch, env.TAG_NAME, true, true)
                    }
                }

                common.conditional_stage('Upload to PyPi', "${env.TAG_NAME}" != "unstable") {
                    container('python37') {
                        withCredentials([
                            usernamePassword(
                                credentialsId: 'iguazio-prod-pypi-credentials',
                                passwordVariable: 'TWINE_PASSWORD',
                                usernameVariable: 'TWINE_USERNAME'
                            )]) {
                                println(common.shellc("pip install twine"))
                                println(common.shellc("MLRUN_VERSION=${dockerTag} make publish-package"))
                        }
                    }
                }

                common.reportStage('update release status') {
                    withCredentials([
                        string(credentialsId: "iguazio-prod-git-user-token", variable: 'GIT_TOKEN')
                    ]) {
                        container('jnlp') {
                            github.update_release_status(gitProject, gitProjectUser, env.TAG_NAME, GIT_TOKEN)
                        }
                    }
                }
            }
        }
    }
}
