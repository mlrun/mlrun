#!/usr/bin/env bash

set -o errexit
set -o pipefail

SCRIPT="$(basename "$0")"

git_owner=mlrun
git_repo=mlrun
git_base_url="https://github.com/${git_owner}/${git_repo}"
git_url="${git_base_url}.git"
user=${V3IO_USERNAME}

USAGE="\
$SCRIPT:
Retrieves updated demos from the mlrun/mlrun GitHub repository.
USAGE: ${SCRIPT} [OPTIONS]
OPTIONS:
  -h|--help   -  Display this message and exit.
  -u|--user   -  Username, which determines the directory to which to copy the
                 retrieved demo files (/v3io/users/<username>) - Iguazio platform.
                 used when --path isn't set.
                 Default: \$V3IO_USERNAME, if set to a non-empty string.
  --mlrun-ver -  The MLRun version for which to get demos; determines the Git
                 branch from which to get the demos e.g. 1.7.0.
                 Default: The version of the installed 'mlrun' package.
  --dry-run   -  Show files to update but don't execute the update.
  --no-backup -  Don't back up the existing demos directory before the update.
                 Default: Back up the existing demos directory to --path parent directory.
  --path      -  demos folder download path e.g. --path=./demos.
                 Default: HOME/demos directory"

# --------------------------------------------------------------------------------------------------------------------------------
# Function for exit due to fatal program error
#   Accepts 1 argument:
#     string containing descriptive error message
# --------------------------------------------------------------------------------------------------------------------------------

error_exit()
{
  echo "${SCRIPT}: ${1:-"Unknown Error"}" 1>&2
  exit 1
}
error_usage()
{
    echo "${SCRIPT}: ${1:-"Unknown Error"}" 1>&2
    echo -e "$USAGE"
    exit 1
}

# --------------------------------------------------------------------------------------------------------------------------------
# Getting arguments from command
# --------------------------------------------------------------------------------------------------------------------------------

while :
do
    case $1 in
        -h | --help) echo -e "$USAGE"; exit 0 ;;
        -u|--user)
            if [ "$2" ]; then
                user=$2
                shift
            # else
                # error_usage "$1: Missing username."
            fi
            ;;
        --user=?*)
            user=${1#*=} # Delete everything up to "=" and assign the remainder.
            ;;
        --mlrun-ver)
            if [ "$2" ]; then
                mlrun_version=$2
                shift
            else
                error_usage "$1: Missing MLRun version."
            fi
            ;;
        --mlrun-ver=?*)
            mlrun_version=${1#*=} # Delete everything up to "=" and assign the remainder.
            ;;
        --mlrun-ver=)         # Handle the case of an empty --mlrun-ver=
            error_usage "$1: Missing MLRun version."
            ;;
        --path=?*)
            demos_dir=${1#*=} # Delete everything up to "=" and assign the remainder.
            ;;
        --dry-run)
            dry_run=1
            ;;
        --no-backup)
            no_backup=1
            ;;
        -*) error_usage "$1: Unknown option."
            ;;
        *) break;
    esac
    shift
done

# --------------------------------------------------------------------------------------------------------------------------------
# Backup old demos and removing demos directory
# --------------------------------------------------------------------------------------------------------------------------------

backup_old_demos(){
    local dest_dir="$1"
    local demos_dir="$2"
    if [ -z "${dry_run}" ]; then
        dt=$(date '+%Y%m%d%H%M%S');
        old_demos_dir="${dest_dir}/demos.old/${dt}"
        echo "Moving existing '${demos_dir}' to ${old_demos_dir}'..."
        mkdir -p "${old_demos_dir}"
        cp -rf "${demos_dir}/." "${old_demos_dir}" || echo "$demos_dir is missing, skipping backup"
        rm -rf "${demos_dir}"
        mkdir -p "${demos_dir}"
    fi
    }

# --------------------------------------------------------------------------------------------------------------------------------
# Function to verify update_demos is present, otherwise grab it
# --------------------------------------------------------------------------------------------------------------------------------

verify_update_demos(){
    local demos_dir="$1"
    local branch="$2"
    if [[ ! -f "${demos_dir}/update_demos.sh" || ${branch}<"v1.7" ]]; then
        echo "update_demos is missing or old, downloading .."
        wget -O "${demos_dir}/update_demos.sh" https://raw.githubusercontent.com/mlrun/mlrun/development/automation/scripts/update_demos.sh
    fi
    }

# --------------------------------------------------------------------------------------------------------------------------------
# Printing flags - dry_run and no_backup
# --------------------------------------------------------------------------------------------------------------------------------

# Don't download new demos only print them
# shellcheck disable=SC2236
if [ ! -z "${dry_run}" ]; then
    echo "Dry run; no files will be copied."
fi
# Don't back up old demos
# shellcheck disable=SC2236
if [ ! -z "${no_backup}" ]; then
    echo "The existing demos directory won't be backed up before the update."
fi

# --------------------------------------------------------------------------------------------------------------------------------
# Detecting demos_dir and dest_dir
# if --path is provided (demos_dir):
#    use it to create detination directory (parent directory)
# else:
#    if --user or V3IO_USERNAME is provided:
#        use it to create demos_dir and dest_dir
#    else: (no user and no v3io_username)
#        use pwd/demos as demos_dir and pwd as dest_dir
# --------------------------------------------------------------------------------------------------------------------------------

current_dir=$(pwd)
folder_name=$(basename "$PWD")
parent_dir=$(dirname "$(pwd)")

# cd to avoid running shell from deleted path
cd
if [ "${demos_dir}" ]; then # means --path is specified
    dest_dir=${demos_dir%/*} # taking parent dir
fi
# Case username isn't provided via command and `V3IO_USERNAME` env variable isn't declared
if [[ -z "${user}" && -z "${demos_dir}" ]]; then
    echo "--user and --path argument are empty, using local path"
    # To support when running inside demos folder or arbitrary folder - add /demos
    if [[ "${folder_name}" == "demos" ]]; then
        demos_dir="${current_dir}"
        dest_dir="${parent_dir}"
    else
        demos_dir="${current_dir}/demos"
        dest_dir="${current_dir}"
    fi
fi
# when --path isn't specified and either V3IO_USERNAME or --user is specified (otherwise case caught above).
if [ -z "${demos_dir}" ]; then
    dest_dir="/v3io/users/${user}"
    demos_dir="${dest_dir}/demos"
fi

# --------------------------------------------------------------------------------------------------------------------------------
# Function to get GitHub repository release index.
# --------------------------------------------------------------------------------------------------------------------------------

get_latest_tag() {
    local mlrun_version="$1"
    local git_owner="$2"
    local git_repo="$3"
    local git_base_url="$4" # Unused in this function but can be useful for future enhancements
    local git_url="$5"

    # Fetch tags from git
    local tags=($(git ls-remote --tags --refs --sort='v:refname' "${git_url}" | awk '{print $2}'))
    # Initialize two empty arrays to hold the two separate lists
    with_rc=()
    without_rc=()
    # Iterate through the list of version strings to split between latest and release
    for version in "${tags[@]}"; do
      tag=${version#refs/tags/}
      if [[ $version == *"rc"* ]]; then
        # If the version string contains "rc," add it to the list with "rc" - only the ones in the form of "something"rcXX
        if [[ $version =~ (^|[^[:alnum:]])rc[0-9]{1,2}$ ]]; then
            with_rc+=("$tag")
        fi
      else
        without_rc+=("$tag")
      fi
    done
    formatted_version=$(echo "$mlrun_version" | sed -E 's/.*([0-9]+\.[0-9]+\.[0-9]+).*$/\1/')
    # finding whether there is a release
    for item in "${without_rc[@]}"; do
      if [[ $item == *"$formatted_version"* ]]; then
        echo "$item"
        return
      fi
    done
    # if release doesn't exists, find matching rc
    formatted_rc=$(echo "$mlrun_version" | sed -E 's/.*rc([0-9]+)?.*/-rc\1/')
    if [ "$formatted_rc" == "$mlrun_version" ]; then # couldn't find rc (mlrun_version is a release with no rc)
      formatted_rc=""
    fi
    all_rcs=()
    for item in "${with_rc[@]}"; do
      if [[ $item == *"$formatted_version"* ]]; then
        all_rcs+=("$item")
      fi
    done
    if [ -z "$all_rcs" ]; then # couldn't find any version, returning latest release
      echo "${without_rc[@]}" | tr ' ' '\n' | sort -r | head -n 1
      return
    else
      # trying to find matching rc
      # case mlrun doesnt have an rc (its a release) and demos doesn't have matching release (fetching latest rc)
      if [ -z "$formatted_rc" ]; then # rc is ""
        echo "${with_rc[*]}" | tr ' ' '\n' | sort -Vr | head -n 1
        return
      fi
      # case mlrun does have an rc - return its matching demos rc
      for item in "${all_rcs[@]}"; do
        if [[ $item == *"$formatted_rc"* ]]; then
          echo "$item"
          return
        fi
      done
      # coldn't find matching rc (mlrun does have an rc but demos doesn't have a matching one) returns latest rc
      echo "${with_rc[*]}" | tr ' ' '\n' | sort -Vr | head -n 1
      return
    fi
    }

# --------------------------------------------------------------------------------------------------------------------------------
# Download tar file to a temporary folder
# --------------------------------------------------------------------------------------------------------------------------------

download_tar_to_temp_dir() {
    local tar_file="$1"
    local temp_dir="$2"
    echo "Downloading : $tar_url ..."
    wget -c "${tar_url}" -O mlrun-demos.tar
    tar -xf mlrun-demos.tar -C "${temp_dir}" --strip-components 1
    rm -rf mlrun-demos.tar
    }
download_tar_gz_to_temp_dir() {
    local tar_file="$1"
    local temp_dir="$2"
    echo "Downloading : $tar_url ..."
    wget -qO- "${tar_url}" | tar xz -C "${temp_dir}" --strip-components 1
    }

# --------------------------------------------------------------------------------------------------------------------------------
# Main script
# backup old demos.
# --------------------------------------------------------------------------------------------------------------------------------

mkdir -p "${demos_dir}"
# Backup demos if needed and deleting demos directory
if [ -z "${no_backup}" ]; then
    backup_old_demos "$dest_dir" "$demos_dir"
else
    # if --dry-run flag isn't set, remove current demos
    if [ -z "${dry_run}" ]; then
        rm -rf "${demos_dir}"
        mkdir -p "${demos_dir}"
    fi
fi

# --------------------------------------------------------------------------------------------------------------------------------
# If --mlrun-ver isn't set, use installed mlrun version.
# When mlrun isn't installed, use 1.7.0.
# --------------------------------------------------------------------------------------------------------------------------------

if [ -z "$mlrun_version" ]; then # mlrun version isn't specified. using installed mlrun version
    pip_mlrun=$(pip show mlrun | grep Version) || :
    if [ -z "${pip_mlrun}" ]; then
        mlrun_version="1.7.0"
        # error_exit "MLRun version not found. Aborting..."
        # no --mlrun-ver and mlrun installed, using mlrun==1.7.0
    else
        echo "Detected MLRun version: ${pip_mlrun}"
        mlrun_version="${pip_mlrun##Version: }"
    fi
fi

# Handling the case mlrun version is 0.0.0+unstable
tag_prefix=`echo "${mlrun_version}" | cut -d . -f1-2`
if [[ "$tag_prefix"=="0.0" ]]; then
    mlrun_version="1.7.0"
fi

echo "Looking for demos with MLRun version - ${mlrun_version}."
if [[ "${mlrun_version}"<"1.7" ]]; then
    git_repo="demos"
fi
latest_tag=$(get_latest_tag "${mlrun_version}" "${git_owner}" "${git_repo}" "${git_base_url}" "${git_url}")
echo "release tag or latest tag : ${latest_tag}"
if [ -z "${latest_tag}" ]; then
     error_exit "Couldn't locate a Git tag with prefix 'v${mlrun_version}.*'."
fi
branch=${latest_tag#refs/tags/}
echo "Using branch ${branch} to download demos"
temp_dir=$(mktemp -d /tmp/temp-get-demos.XXXXXXXXXX)
# demos introduced to mlrun in 1.7.0
if [[ "${branch}">"v1.7" ]]; then
    tar_url="${git_base_url}/releases/download/${branch}/mlrun-demos.tar"
    download_tar_to_temp_dir "$tar_url" "$temp_dir"
    verify_update_demos "${temp_dir}" "${branch}"
else
    git_repo="demos"
    git_base_url="https://github.com/${git_owner}/${git_repo}"
    tar_url="${git_base_url}/archive/${branch}.tar.gz"
    download_tar_gz_to_temp_dir "$tar_url" "$temp_dir"
    verify_update_demos "${temp_dir}" "${branch}"
fi
if [ -z "${dry_run}" ]; then
    echo "copy files from ${temp_dir} to ${demos_dir}"
    cp -rf "$temp_dir/." "$demos_dir"
else
    echo "Identified the following files to copy to '${dest_dir}':"
    cd ${temp_dir}
    echo "$(ls -a)"
fi
