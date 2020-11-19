
@Library('pipelinex@_refactor_github') _

label = "${UUID.randomUUID().toString()}"
git_project = "mlrun"
git_project_user = "mlrun"
git_deploy_user_token = "iguazio-prod-git-user-token"
git_deploy_user_private_key = "iguazio-prod-git-user-private-key"
git_mlrun_ui_project = "ui"

podTemplate(label: "${git_project}-${label}", inheritFrom: "docker-python") {
    node("${git_project}-${label}") {
        pipelinex = library(identifier: 'pipelinex@_refactor_github', retriever: modernSCM(
                [$class       : 'GitSCMSource',
                 credentialsId: git_deploy_user_private_key,
                 remote       : "git@github.com:iguazio/pipelinex.git"])).com.iguazio.pipelinex
        common.notify_slack {
            withCredentials([
                    string(credentialsId: git_deploy_user_token, variable: 'GIT_TOKEN')
            ]) {
                def mlrun_github_client = new Githubc(git_project_user, git_project, GIT_TOKEN, env.TAG_NAME, this)
                def ui_github_client = new Githubc(git_project_user, git_mlrun_ui_project, GIT_TOKEN, this)

                mlrun_github_client.releaseCi(true) {
                    container('docker-python') {
                        stage("build ${git_project}/api in dood") {
                            dir("${Githubc.BUILD_FOLDER}/src/github.com/${git_project_user}/${git_project}") {
                                println("TEST")
                                // println(common.shellc("MLRUN_VERSION=${mlrun_github_client.tag.docker} make api"))
                            }
                        }

                        // dockerx.images_push_multi_registries(["${git_project}/mlrun-api:${mlrun_github_client.tag.docker}"], [pipelinex.DockerRepo.ARTIFACTORY_IGUAZIO, pipelinex.DockerRepo.MLRUN_DOCKER_HUB, pipelinex.DockerRepo.MLRUN_QUAY_IO])

                        stage("build ${git_project}/mlrun in dood") {
                            dir("${Githubc.BUILD_FOLDER}/src/github.com/${git_project_user}/${git_project}") {
                                // println(common.shellc("MLRUN_VERSION=${mlrun_github_client.tag.docker} make mlrun"))
                                println("TEST")
                            }
                        }

                        // dockerx.images_push_multi_registries(["${git_project}/mlrun:${mlrun_github_client.tag.docker}"], [pipelinex.DockerRepo.ARTIFACTORY_IGUAZIO, pipelinex.DockerRepo.MLRUN_DOCKER_HUB, pipelinex.DockerRepo.MLRUN_QUAY_IO])

                        stage("build ${git_project}/jupyter in dood") {
                            dir("${Githubc.BUILD_FOLDER}/src/github.com/${git_project_user}/${git_project}") {
                                // println(common.shellc("MLRUN_VERSION=${mlrun_github_client.tag.docker} make jupyter"))
                                println("TEST")
                            }
                        }

                        // dockerx.images_push_multi_registries(["${git_project}/jupyter:${mlrun_github_client.tag.docker}"], [pipelinex.DockerRepo.ARTIFACTORY_IGUAZIO, pipelinex.DockerRepo.MLRUN_DOCKER_HUB, pipelinex.DockerRepo.MLRUN_QUAY_IO])

                        stage("build ${git_project}/base in dood") {
                            dir("${Githubc.BUILD_FOLDER}/src/github.com/${git_project_user}/${git_project}") {
                                // println(common.shellc("MLRUN_VERSION=${mlrun_github_client.tag.docker} make base"))
                                println("TEST")
                            }
                        }

                        // dockerx.images_push_multi_registries(["${git_project}/ml-base:${mlrun_github_client.tag.docker}"], [pipelinex.DockerRepo.ARTIFACTORY_IGUAZIO, pipelinex.DockerRepo.MLRUN_DOCKER_HUB, pipelinex.DockerRepo.MLRUN_QUAY_IO])

                        stage("build ${git_project}/base-legacy in dood") {
                            dir("${Githubc.BUILD_FOLDER}/src/github.com/${git_project_user}/${git_project}") {
                                // println(common.shellc("MLRUN_VERSION=${mlrun_github_client.tag.docker} make base-legacy"))
                                println("TEST")
                            }
                        }

                        // dockerx.images_push_multi_registries(["${git_project}/ml-base:${mlrun_github_client.tag.docker}-py36"], [pipelinex.DockerRepo.ARTIFACTORY_IGUAZIO, pipelinex.DockerRepo.MLRUN_DOCKER_HUB, pipelinex.DockerRepo.MLRUN_QUAY_IO])

                        stage("build ${git_project}/models in dood") {
                            dir("${Githubc.BUILD_FOLDER}/src/github.com/${git_project_user}/${git_project}") {
                                // println(common.shellc("MLRUN_VERSION=${mlrun_github_client.tag.docker} make models"))
                                println("TEST")
                            }
                        }

                        // dockerx.images_push_multi_registries(["${git_project}/ml-models:${mlrun_github_client.tag.docker}"], [pipelinex.DockerRepo.ARTIFACTORY_IGUAZIO, pipelinex.DockerRepo.MLRUN_DOCKER_HUB, pipelinex.DockerRepo.MLRUN_QUAY_IO])

                        stage("build ${git_project}/models-legacy in dood") {
                            dir("${Githubc.BUILD_FOLDER}/src/github.com/${git_project_user}/${git_project}") {
                                // println(common.shellc("MLRUN_VERSION=${mlrun_github_client.tag.docker} make models-legacy"))
                                println("TEST")
                            }
                        }

                        // dockerx.images_push_multi_registries(["${git_project}/ml-models:${mlrun_github_client.tag.docker}-py36"], [pipelinex.DockerRepo.ARTIFACTORY_IGUAZIO, pipelinex.DockerRepo.MLRUN_DOCKER_HUB, pipelinex.DockerRepo.MLRUN_QUAY_IO])

                        stage("build ${git_project}/models-gpu in dood") {
                            dir("${Githubc.BUILD_FOLDER}/src/github.com/${git_project_user}/${git_project}") {
                                // println(common.shellc("MLRUN_VERSION=${mlrun_github_client.tag.docker} make models-gpu"))
                                println("TEST")
                            }
                        }

                        // dockerx.images_push_multi_registries(["${git_project}/ml-models-gpu:${mlrun_github_client.tag.docker}"], [pipelinex.DockerRepo.ARTIFACTORY_IGUAZIO, pipelinex.DockerRepo.MLRUN_DOCKER_HUB, pipelinex.DockerRepo.MLRUN_QUAY_IO])

                        stage("build ${git_project}/models-gpu-legacy in dood") {
                            dir("${Githubc.BUILD_FOLDER}/src/github.com/${git_project_user}/${git_project}") {
                                // println(common.shellc("MLRUN_VERSION=${mlrun_github_client.tag.docker} make models-gpu-legacy"))
                                println("TEST")
                            }
                        }

                        // dockerx.images_push_multi_registries(["${git_project}/ml-models-gpu:${mlrun_github_client.tag.docker}-py36"], [pipelinex.DockerRepo.ARTIFACTORY_IGUAZIO, pipelinex.DockerRepo.MLRUN_DOCKER_HUB, pipelinex.DockerRepo.MLRUN_QUAY_IO])
                    }
                    container('jnlp') {
                        common.conditional_stage('Create mlrun/ui release', "${mlrun_github_client.tag.toString()}" != "unstable") {
                            def source_branch = mlrun_github_client.getReleasecommittish()
                            
                            print("source branch is: ${source_branch}, using this as source for mlrun/ui")
                            if (!source_branch) {
                                error("Could not get source branch from tag")
                            }

                            ui_github_client.createRelease(source_branch, mlrun_github_client.tag.toString(), true)

                            // github.wait_for_release(
                            //         git_mlrun_ui_project,
                            //         git_project_user,
                            //         "${github.TAG_VERSION}",
                            //         GIT_TOKEN
                            // )
                        }
                    }

                    // common.conditional_stage('Upload to PyPi', "${mlrun_github_client.tag.toString()}" != "unstable") {
                    //     container('python37') {
                    //         withCredentials([
                    //             usernamePassword(credentialsId: "iguazio-prod-pypi-credentials",
                    //                                 passwordVariable: 'TWINE_PASSWORD',
                    //                                 usernameVariable: 'TWINE_USERNAME')]) {
                    //             dir("${Githubc.BUILD_FOLDER}/src/github.com/${git_project_user}/${git_project}") {
                    //                 println(common.shellc("pip install twine"))
                    //                 println(common.shellc("MLRUN_VERSION=${mlrun_github_client.tag.docker} make publish-package"))
                    //             }
                    //         }
                    //     }
                    // }
                }
            }
        }
    }
}
