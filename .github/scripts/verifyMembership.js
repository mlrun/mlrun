const core = require("@actions/core");
const github = require("@actions/github");

async function run() {
    try {
        const token = process.env.GITHUB_TOKEN;
        const orgName = process.env.ORG_NAME;

        if (!token || !orgName) {
            core.setFailed("❌ GITHUB_TOKEN or ORG_NAME is not set.");
            return;
        }

        const octokit = github.getOctokit(token);

        // Extract PR author from GitHub context
        const prAuthor = github.context.payload.pull_request?.user?.login;
        if (!prAuthor) {
            core.setFailed("❌ Could not determine the PR author.");
            return;
        }

        switch (prAuthor) {
            case "dependabot[bot]":
                console.log("🤖 Skipping verification for dependabot PRs.");
                return;
            case orgName:
                console.log(`🔍 Skipping verification for PRs created by ${orgName}.`);
                return;
            default:
                console.log(`🔍 Verifying membership for user ${prAuthor} in org ${orgName}.`);
                try {
                    const response = await octokit.rest.orgs.getMembershipForUser({
                        org: orgName,
                        username: prAuthor,
                    });

                    if (response.status === 200 && response.data.state === "active") {
                        console.log(`✅ User ${prAuthor} is a verified member of ${orgName}.`);
                    } else {
                        core.setFailed(`❌ User ${prAuthor} is NOT a member of ${orgName}.`);
                    }
                } catch (error) {
                    core.setFailed(`❌ API Error: ${error.message}`);
                }
        }
    } catch (error) {
        core.setFailed(`❌ Unexpected Error: ${error.message}`);
    }
}

run();
