#!/usr/bin/env bash
# Shared GitHub App token generation and refresh utilities.
#
# Required environment variables:
#   GH_APP_ID          - GitHub App ID
#   GH_APP_PRIVATE_KEY - GitHub App private key (PEM)
#
# Usage:
#   source .github/scripts/github-app-token.sh
#   TOKEN=$(generate_installation_token "mlrun" "mlrun,private-system-tests")
#
#   # Or start a background daemon that keeps env.yml updated:
#   start_token_refresh_daemon "/tmp/.gh-token" "tests/system/env.yml" "MLRUN_SYSTEM_TESTS_GIT_TOKEN" 3300 "mlrun" "mlrun,private-system-tests"

generate_jwt() {
  local app_id="${GH_APP_ID:?GH_APP_ID must be set}"
  local private_key="${GH_APP_PRIVATE_KEY:?GH_APP_PRIVATE_KEY must be set}"
  local now=$(date +%s)
  local iat=$((now - 60))
  local exp=$((now + 540)) # 9 minutes (max 10)

  local header=$(echo -n '{"alg":"RS256","typ":"JWT"}' \
    | openssl base64 -e -A | tr '+/' '-_' | tr -d '=')
  local payload=$(echo -n "{\"iat\":${iat},\"exp\":${exp},\"iss\":\"${app_id}\"}" \
    | openssl base64 -e -A | tr '+/' '-_' | tr -d '=')
  local unsigned="${header}.${payload}"
  local signature=$(echo -n "${unsigned}" \
    | openssl dgst -sha256 -sign <(echo "${private_key}") \
    | openssl base64 -e -A | tr '+/' '-_' | tr -d '=')
  echo "${unsigned}.${signature}"
}

generate_installation_token() {
  local owner="${1:?owner required}"
  local repos="${2:-}"

  local jwt=$(generate_jwt)
  local installation_id=$(curl -sf \
    -H "Authorization: Bearer ${jwt}" \
    -H "Accept: application/vnd.github+json" \
    "https://api.github.com/app/installations" \
    | jq -r ".[] | select(.account.login==\"${owner}\") | .id")

  if [ -z "${installation_id}" ]; then
    echo "ERROR: Could not find installation for owner '${owner}'" >&2
    return 1
  fi

  local data='{}'
  if [ -n "${repos}" ]; then
    data=$(echo "${repos}" | jq -R 'split(",") | map(gsub("^\\s+|\\s+$";"")) | {"repositories": .}')
  fi

  local token=$(curl -sf -X POST \
    -H "Authorization: Bearer ${jwt}" \
    -H "Accept: application/vnd.github+json" \
    "https://api.github.com/app/installations/${installation_id}/access_tokens" \
    -d "${data}" \
    | jq -r '.token')

  if [ -z "${token}" ] || [ "${token}" = "null" ]; then
    echo "ERROR: Failed to generate installation token" >&2
    return 1
  fi

  echo "${token}"
}

# Start a background daemon that refreshes the token periodically.
# It writes the fresh token to a file and optionally updates a YAML env file on disk
# so that test classes picking up env.yml mid-run get a valid token.
#
# Args:
#   $1 - token_file        Path to write the raw token (e.g. /tmp/.gh-token)
#   $2 - env_yml_file      Path to env.yml to update (empty string to skip)
#   $3 - env_var_name       Key name in env.yml (e.g. MLRUN_SYSTEM_TESTS_GIT_TOKEN)
#   $4 - refresh_interval  Seconds between refreshes (default 3300 = 55 min)
#   $5 - owner             GitHub org (e.g. mlrun)
#   $6 - repos             Comma-separated repos (e.g. mlrun,private-system-tests)
#
# Returns: prints the daemon PID
start_token_refresh_daemon() {
  local token_file="${1:?token_file required}"
  local env_yml_file="${2:-}"
  local env_var_name="${3:-MLRUN_SYSTEM_TESTS_GIT_TOKEN}"
  local refresh_interval="${4:-3300}"
  local owner="${5:-${GITHUB_REPOSITORY_OWNER}}"
  local repos="${6:-}"

  local initial_token=$(generate_installation_token "${owner}" "${repos}")
  echo "${initial_token}" > "${token_file}"

  if [ -n "${env_yml_file}" ] && [ -f "${env_yml_file}" ]; then
    _update_env_yml "${env_yml_file}" "${env_var_name}" "${initial_token}"
  fi

  (
    while true; do
      sleep "${refresh_interval}"
      local new_token=$(generate_installation_token "${owner}" "${repos}" 2>/dev/null)
      if [ -n "${new_token}" ] && [ "${new_token}" != "null" ]; then
        echo "${new_token}" > "${token_file}"
        if [ -n "${env_yml_file}" ] && [ -f "${env_yml_file}" ]; then
          _update_env_yml "${env_yml_file}" "${env_var_name}" "${new_token}"
        fi
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Token refreshed" >&2
      else
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) WARNING: Token refresh failed, will retry" >&2
      fi
    done
  ) &
  echo $!
}

_update_env_yml() {
  local file="$1"
  local key="$2"
  local value="$3"

  if grep -q "^${key}:" "${file}" 2>/dev/null; then
    sed -i "s|^${key}:.*|${key}: ${value}|" "${file}"
  else
    echo "${key}: ${value}" >> "${file}"
  fi
}
