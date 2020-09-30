
@Library('pipelinex@_refactor_github') _

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
                mlrun_github = new Githubc(git_project_user, git_project, GIT_TOKEN)

                println(mlrun_github.getReleaseId('v0.5.2'))

            }
        }
    }
}
