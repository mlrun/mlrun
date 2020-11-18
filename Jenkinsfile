
label = "${UUID.randomUUID().toString()}"
git_project = "mlrun"
git_project_user = "mlrun"
git_project_upstream_user = "mlrun"
git_deploy_user = "iguazio-prod-git-user"
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
                def github_client = new Githubc(git_project_user, git_project, GIT_TOKEN, env.TAG_NAME, this)
                github_client.releaseCi(true) {
                    container('docker-python') {
                        stage("build ${git_project}/api in dood") {
                            dir("${env.BUILD_FOLDER}/src/github.com/${git_project_user}/${git_project}") {
                                println(common.shellc("MLRUN_VERSION=${github_client.tag.docker} make api"))
                            }
                        }

                        // dockerx.images_push_multi_registries(["${git_project}/mlrun-api:${github_client.tag.docker}"], [pipelinex.DockerRepo.ARTIFACTORY_IGUAZIO, pipelinex.DockerRepo.MLRUN_DOCKER_HUB, pipelinex.DockerRepo.MLRUN_QUAY_IO])

                        stage("build ${git_project}/mlrun in dood") {
                            dir("${env.BUILD_FOLDER}/src/github.com/${git_project_upstream_user}/${git_project}") {
                                println(common.shellc("MLRUN_VERSION=${github_client.tag.docker} make mlrun"))
                            }
                        }

                        // dockerx.images_push_multi_registries(["${git_project}/mlrun:${github_client.tag.docker}"], [pipelinex.DockerRepo.ARTIFACTORY_IGUAZIO, pipelinex.DockerRepo.MLRUN_DOCKER_HUB, pipelinex.DockerRepo.MLRUN_QUAY_IO])

                        stage("build ${git_project}/jupyter in dood") {
                            dir("${env.BUILD_FOLDER}/src/github.com/${git_project_upstream_user}/${git_project}") {
                                println(common.shellc("MLRUN_VERSION=${github_client.tag.docker} make jupyter"))
                            }
                        }

                        // dockerx.images_push_multi_registries(["${git_project}/jupyter:${github_client.tag.docker}"], [pipelinex.DockerRepo.ARTIFACTORY_IGUAZIO, pipelinex.DockerRepo.MLRUN_DOCKER_HUB, pipelinex.DockerRepo.MLRUN_QUAY_IO])

                        stage("build ${git_project}/base in dood") {
                            dir("${env.BUILD_FOLDER}/src/github.com/${git_project_upstream_user}/${git_project}") {
                                println(common.shellc("MLRUN_VERSION=${github_client.tag.docker} make base"))
                            }
                        }

                        // dockerx.images_push_multi_registries(["${git_project}/ml-base:${github_client.tag.docker}"], [pipelinex.DockerRepo.ARTIFACTORY_IGUAZIO, pipelinex.DockerRepo.MLRUN_DOCKER_HUB, pipelinex.DockerRepo.MLRUN_QUAY_IO])

                        stage("build ${git_project}/base-legacy in dood") {
                            dir("${env.BUILD_FOLDER}/src/github.com/${git_project_upstream_user}/${git_project}") {
                                println(common.shellc("MLRUN_VERSION=${github_client.tag.docker} make base-legacy"))
                            }
                        }

                        // dockerx.images_push_multi_registries(["${git_project}/ml-base:${github_client.tag.docker}-py36"], [pipelinex.DockerRepo.ARTIFACTORY_IGUAZIO, pipelinex.DockerRepo.MLRUN_DOCKER_HUB, pipelinex.DockerRepo.MLRUN_QUAY_IO])

                        stage("build ${git_project}/models in dood") {
                            dir("${env.BUILD_FOLDER}/src/github.com/${git_project_upstream_user}/${git_project}") {
                                println(common.shellc("MLRUN_VERSION=${github_client.tag.docker} make models"))
                            }
                        }

                        // dockerx.images_push_multi_registries(["${git_project}/ml-models:${github_client.tag.docker}"], [pipelinex.DockerRepo.ARTIFACTORY_IGUAZIO, pipelinex.DockerRepo.MLRUN_DOCKER_HUB, pipelinex.DockerRepo.MLRUN_QUAY_IO])

                        stage("build ${git_project}/models-legacy in dood") {
                            dir("${env.BUILD_FOLDER}/src/github.com/${git_project_upstream_user}/${git_project}") {
                                println(common.shellc("MLRUN_VERSION=${github_client.tag.docker} make models-legacy"))
                            }
                        }

                        // dockerx.images_push_multi_registries(["${git_project}/ml-models:${github_client.tag.docker}-py36"], [pipelinex.DockerRepo.ARTIFACTORY_IGUAZIO, pipelinex.DockerRepo.MLRUN_DOCKER_HUB, pipelinex.DockerRepo.MLRUN_QUAY_IO])

                        stage("build ${git_project}/models-gpu in dood") {
                            dir("${env.BUILD_FOLDER}/src/github.com/${git_project_upstream_user}/${git_project}") {
                                println(common.shellc("MLRUN_VERSION=${github_client.tag.docker} make models-gpu"))
                            }
                        }

                        // dockerx.images_push_multi_registries(["${git_project}/ml-models-gpu:${github_client.tag.docker}"], [pipelinex.DockerRepo.ARTIFACTORY_IGUAZIO, pipelinex.DockerRepo.MLRUN_DOCKER_HUB, pipelinex.DockerRepo.MLRUN_QUAY_IO])

                        stage("build ${git_project}/models-gpu-legacy in dood") {
                            dir("${env.BUILD_FOLDER}/src/github.com/${git_project_upstream_user}/${git_project}") {
                                println(common.shellc("MLRUN_VERSION=${github_client.tag.docker} make models-gpu-legacy"))
                            }
                        }

                        // dockerx.images_push_multi_registries(["${git_project}/ml-models-gpu:${github_client.tag.docker}-py36"], [pipelinex.DockerRepo.ARTIFACTORY_IGUAZIO, pipelinex.DockerRepo.MLRUN_DOCKER_HUB, pipelinex.DockerRepo.MLRUN_QUAY_IO])
                    }
                    container('jnlp') {
                        common.conditional_stage('Create mlrun/ui release', "${github.TAG_VERSION}" != "unstable") {
                            def source_branch = github_client.getReleasecommittish(github_client.tag.toString())
                            
                            print("source branch is: ${source_branch}, using this as source for mlrun/ui")
                            if (!source_branch) {
                                error("Could not get source branch from tag")
                            }

                            // github.create_prerelease(
                            //         git_mlrun_ui_project,
                            //         git_project_upstream_user,
                            //         "${github.TAG_VERSION}",
                            //         GIT_TOKEN,
                            //         "${source_branch}"
                            // )
                            // github.wait_for_release(
                            //         git_mlrun_ui_project,
                            //         git_project_upstream_user,
                            //         "${github.TAG_VERSION}",
                            //         GIT_TOKEN
                            // )
                        }
                    }

                    // common.conditional_stage('Upload to PyPi', "${github.TAG_VERSION}" != "unstable") {
                    //     container('python37') {
                    //         withCredentials([
                    //             usernamePassword(credentialsId: "iguazio-prod-pypi-credentials",
                    //                                 passwordVariable: 'TWINE_PASSWORD',
                    //                                 usernameVariable: 'TWINE_USERNAME')]) {
                    //             dir("${env.BUILD_FOLDER}/src/github.com/${git_project_upstream_user}/${git_project}") {
                    //                 println(common.shellc("pip install twine"))
                    //                 println(common.shellc("MLRUN_VERSION=${github_client.tag.docker} make publish-package"))
                    //             }
                    //         }
                    //     }
                    // }
                }
            }
        }
    }
}
