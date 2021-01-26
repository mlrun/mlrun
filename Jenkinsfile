
label = "${UUID.randomUUID().toString()}"
git_project = "mlrun"
git_project_user = "mlrun"
git_project_upstream_user = "mlrun"
git_deploy_user = "iguazio-prod-git-user"
git_deploy_user_token = "iguazio-prod-git-user-token"
git_deploy_user_private_key = "iguazio-prod-git-user-private-key"
git_mlrun_ui_project = "ui"

podTemplate(label: "${git_project}-${label}", inheritFrom: "jnlp-docker-golang-python37") {
    node("${git_project}-${label}") {
        pipelinex = library(identifier: 'pipelinex@development', retriever: modernSCM(
                [$class       : 'GitSCMSource',
                 credentialsId: git_deploy_user_private_key,
                 remote       : "git@github.com:iguazio/pipelinex.git"])).com.iguazio.pipelinex
        common.notify_slack {
            withCredentials([
                    string(credentialsId: git_deploy_user_token, variable: 'GIT_TOKEN')
            ]) {
                github.release(git_deploy_user, git_project, git_project_user, git_project_upstream_user, true, GIT_TOKEN) {
                    container('docker-python') {
                        stage("build ${git_project}/api in dood") {
                            dir("${github.BUILD_FOLDER}/src/github.com/${git_project_upstream_user}/${git_project}") {
                                println(common.shellc("MLRUN_VERSION=${github.DOCKER_TAG_VERSION} make api"))
                            }
                        }

                        dockerx.images_push_multi_registries(["${git_project}/mlrun-api:${github.DOCKER_TAG_VERSION}"], [pipelinex.DockerRepo.ARTIFACTORY_IGUAZIO, pipelinex.DockerRepo.MLRUN_DOCKER_HUB, pipelinex.DockerRepo.MLRUN_QUAY_IO])

                        stage("build ${git_project}/mlrun in dood") {
                            dir("${github.BUILD_FOLDER}/src/github.com/${git_project_upstream_user}/${git_project}") {
                                println(common.shellc("MLRUN_VERSION=${github.DOCKER_TAG_VERSION} make mlrun"))
                            }
                        }

                        dockerx.images_push_multi_registries(["${git_project}/mlrun:${github.DOCKER_TAG_VERSION}"], [pipelinex.DockerRepo.ARTIFACTORY_IGUAZIO, pipelinex.DockerRepo.MLRUN_DOCKER_HUB, pipelinex.DockerRepo.MLRUN_QUAY_IO])

                        stage("build ${git_project}/jupyter in dood") {
                            dir("${github.BUILD_FOLDER}/src/github.com/${git_project_upstream_user}/${git_project}") {
                                println(common.shellc("MLRUN_VERSION=${github.DOCKER_TAG_VERSION} make jupyter"))
                            }
                        }

                        dockerx.images_push_multi_registries(["${git_project}/jupyter:${github.DOCKER_TAG_VERSION}"], [pipelinex.DockerRepo.ARTIFACTORY_IGUAZIO, pipelinex.DockerRepo.MLRUN_DOCKER_HUB, pipelinex.DockerRepo.MLRUN_QUAY_IO])

                        stage("build ${git_project}/base in dood") {
                            dir("${github.BUILD_FOLDER}/src/github.com/${git_project_upstream_user}/${git_project}") {
                                println(common.shellc("MLRUN_VERSION=${github.DOCKER_TAG_VERSION} make base"))
                            }
                        }

                        dockerx.images_push_multi_registries(["${git_project}/ml-base:${github.DOCKER_TAG_VERSION}"], [pipelinex.DockerRepo.ARTIFACTORY_IGUAZIO, pipelinex.DockerRepo.MLRUN_DOCKER_HUB, pipelinex.DockerRepo.MLRUN_QUAY_IO])

                        stage("build ${git_project}/base-legacy in dood") {
                            dir("${github.BUILD_FOLDER}/src/github.com/${git_project_upstream_user}/${git_project}") {
                                println(common.shellc("MLRUN_VERSION=${github.DOCKER_TAG_VERSION} make base-legacy"))
                            }
                        }

                        dockerx.images_push_multi_registries(["${git_project}/ml-base:${github.DOCKER_TAG_VERSION}-py36"], [pipelinex.DockerRepo.ARTIFACTORY_IGUAZIO, pipelinex.DockerRepo.MLRUN_DOCKER_HUB, pipelinex.DockerRepo.MLRUN_QUAY_IO])

                        stage("build ${git_project}/models in dood") {
                            dir("${github.BUILD_FOLDER}/src/github.com/${git_project_upstream_user}/${git_project}") {
                                println(common.shellc("MLRUN_VERSION=${github.DOCKER_TAG_VERSION} make models"))
                            }
                        }

                        dockerx.images_push_multi_registries(["${git_project}/ml-models:${github.DOCKER_TAG_VERSION}"], [pipelinex.DockerRepo.ARTIFACTORY_IGUAZIO, pipelinex.DockerRepo.MLRUN_DOCKER_HUB, pipelinex.DockerRepo.MLRUN_QUAY_IO])

                        stage("build ${git_project}/models-legacy in dood") {
                            dir("${github.BUILD_FOLDER}/src/github.com/${git_project_upstream_user}/${git_project}") {
                                println(common.shellc("MLRUN_VERSION=${github.DOCKER_TAG_VERSION} make models-legacy"))
                            }
                        }

                        dockerx.images_push_multi_registries(["${git_project}/ml-models:${github.DOCKER_TAG_VERSION}-py36"], [pipelinex.DockerRepo.ARTIFACTORY_IGUAZIO, pipelinex.DockerRepo.MLRUN_DOCKER_HUB, pipelinex.DockerRepo.MLRUN_QUAY_IO])

                        stage("build ${git_project}/models-gpu in dood") {
                            dir("${github.BUILD_FOLDER}/src/github.com/${git_project_upstream_user}/${git_project}") {
                                println(common.shellc("MLRUN_VERSION=${github.DOCKER_TAG_VERSION} make models-gpu"))
                            }
                        }

                        dockerx.images_push_multi_registries(["${git_project}/ml-models-gpu:${github.DOCKER_TAG_VERSION}"], [pipelinex.DockerRepo.ARTIFACTORY_IGUAZIO, pipelinex.DockerRepo.MLRUN_DOCKER_HUB, pipelinex.DockerRepo.MLRUN_QUAY_IO])

                        stage("build ${git_project}/models-gpu-legacy in dood") {
                            dir("${github.BUILD_FOLDER}/src/github.com/${git_project_upstream_user}/${git_project}") {
                                println(common.shellc("MLRUN_VERSION=${github.DOCKER_TAG_VERSION} make models-gpu-legacy"))
                            }
                        }

                        dockerx.images_push_multi_registries(["${git_project}/ml-models-gpu:${github.DOCKER_TAG_VERSION}-py36"], [pipelinex.DockerRepo.ARTIFACTORY_IGUAZIO, pipelinex.DockerRepo.MLRUN_DOCKER_HUB, pipelinex.DockerRepo.MLRUN_QUAY_IO])
                    }
                    container('jnlp') {
                        common.conditional_stage('Create mlrun/ui release', "${github.TAG_VERSION}" != "unstable") {
                            def source_branch = github.get_release_commitish(
                                                            git_project,
                                                            git_project_upstream_user,
                                                            "${github.TAG_VERSION}",
                                                            GIT_TOKEN
                            )
                            print("source branch is: ${source_branch}, using this as source for mlrun/ui")
                            if (!source_branch) {
                                error("Could not get source branch from tag ${github.TAG_VERSION} via git command")
                            }
                            github.create_prerelease(
                                    git_mlrun_ui_project,
                                    git_project_upstream_user,
                                    "${github.TAG_VERSION}",
                                    GIT_TOKEN,
                                    "${source_branch}"
                            )
                            github.wait_for_release(
                                    git_mlrun_ui_project,
                                    git_project_upstream_user,
                                    "${github.TAG_VERSION}",
                                    GIT_TOKEN
                            )
                        }
                    }

                    common.conditional_stage('Upload to PyPi', "${github.TAG_VERSION}" != "unstable") {
                        container('python37') {
                            withCredentials([
                                usernamePassword(credentialsId: "iguazio-prod-pypi-credentials",
                                                    passwordVariable: 'TWINE_PASSWORD',
                                                    usernameVariable: 'TWINE_USERNAME')]) {
                                dir("${github.BUILD_FOLDER}/src/github.com/${git_project_upstream_user}/${git_project}") {
                                    println(common.shellc("pip install twine"))
                                    println(common.shellc("MLRUN_VERSION=${github.DOCKER_TAG_VERSION} make publish-package"))
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
