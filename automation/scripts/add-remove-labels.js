// Copyright 2026 Iguazio
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Reusable label management for GitHub Actions workflows.
// Usage with actions/github-script:
//
//   const addRemoveLabels = require('./automation/scripts/add-remove-labels.js')
//   await addRemoveLabels({
//     github, context,
//     prNumber: 123,
//     labelsToAdd: [{ name: 'Smoke tests: Pass', color: '0e8a16' }],
//     labelsToRemove: ['Smoke tests: Fail', 'Smoke tests: Unknown']
//   })

module.exports = async ({ github, context, prNumber, labelsToAdd = [], labelsToRemove = [] }) => {
  const owner = context.repo.owner
  const repo = context.repo.repo

  // Get current labels on the PR
  const { data: currentLabels } = await github.rest.issues.listLabelsOnIssue({
    owner,
    repo,
    issue_number: prNumber,
  })
  const currentNames = currentLabels.map((l) => l.name)

  // Remove labels, skipping any that will be added back
  const addNames = new Set(labelsToAdd.map((l) => l.name))
  for (const label of labelsToRemove) {
    if (addNames.has(label)) {
      continue
    }
    if (!currentNames.includes(label)) {
      continue
    }
    try {
      await github.rest.issues.removeLabel({
        owner,
        repo,
        issue_number: prNumber,
        name: label,
      })
      console.log(`Removed label: ${label}`)
    } catch (error) {
      if (error.status === 404) {
        console.log(`Label '${label}' not found on PR, skipping removal`)
      } else {
        throw error
      }
    }
  }

  // Add labels (ensure they exist in the repo first)
  for (const { name, color } of labelsToAdd) {
    if (currentNames.includes(name)) {
      console.log(`Label '${name}' already set, skipping`)
      continue
    }

    // Create or update the label in the repo
    try {
      await github.rest.issues.updateLabel({
        owner,
        repo,
        name,
        color,
      })
    } catch {
      await github.rest.issues.createLabel({
        owner,
        repo,
        name,
        color,
      })
    }

    await github.rest.issues.addLabels({
      owner,
      repo,
      issue_number: prNumber,
      labels: [name],
    })
    console.log(`Added label: ${name}`)
  }
}
