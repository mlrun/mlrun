@Library('pipelinex@development') _
import com.iguazio.pipelinex.DockerRepo

workDir = '/home/jenkins'
podLabel = 'mlrun-release'
gitProject = 'mlrun'
gitProjectUser = 'mlrun'
gitProjectUI = 'ui'
source_branch = 'master'
dockerTag = env.TAG_NAME.replaceFirst(/^v/, '')

properties_args = [
    parameters([
        booleanParam(defaultValue: true, description: '', name: 'release_mlrun_ui'),
    ]),
]

properties(properties_args)

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
                }

                container('jnlp') {
                    common.conditional_stage('Create mlrun/ui release', true) {
                        println("Source branch is: ${source_branch}, using this as source for ${gitProject}/${gitProjectUI}")
                        println("You are responsible to make sure that this branch exists in ${gitProject}/${gitProjectUI}!")


                        def mlrun_github_client = new Githubc(gitProjectUser, gitProject, GIT_TOKEN, env.TAG_NAME, this)
                        def ui_github_client = new Githubc(gitProjectUser, gitProjectUI, GIT_TOKEN, this)
                        def source_branch = mlrun_github_client.getReleasecommittish()


                        if (!source_branch) {
                            error("Could not get source branch from tag")
                        }

                        ui_github_client.createRelease(source_branch, env.TAG_NAME, true, true)
                    }
                }

            }
        }
    }
}
