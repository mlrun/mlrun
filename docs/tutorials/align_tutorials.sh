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
Retrieves updated tutorials from the mlrun/mlrun GitHub repository.
USAGE: ${SCRIPT} [OPTIONS]
OPTIONS:
  -h|--help   -  Display this message and exit.
  -b|--branch -  mlrun/mlrun repository branch name, can't be set together with --mlrun-ver.
  -u|--user   -  Username, which determines the directory to which to copy the
                 retrieved demo files (/v3io/users/<username>) - Iguazio platform.
                 Default: \$V3IO_USERNAME, if set to a non-empty string.
  --mlrun-ver -  The MLRun version for which to get tutorials; determines the Git
                 branch from which to get the tutorials, unless -b|--branch is set.
                 Default: The version of the installed 'mlrun' package.
  --dry-run   -  Show files to update but don't execute the update.
  --no-backup -  Don't back up the existing tutorials directory before the update.
                 Default: Back up the existing tutorials directory to --path parent directory.
  --path      -  tutorials folder download path e.g. --path=./tutorials.
                 Default: PWD/tutorials directory"

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
        -b|--branch)
            if [ "$2" ]; then
                branch=$2
                shift
            else
                error_usage "$1: Missing branch name."
            fi
            ;;
        --branch=?*)
            branch=${1#*=} # Delete everything up to "=" and assign the remainder.
            ;;
        --branch=)         # Handle the case of an empty --branch=
            error_usage "$1: Missing branch name."
            ;;
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
        # --user=)         # Handle the case of an empty --user=
            # error_usage "$1: Missing username."
            # ;;
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
            tutorials_dir=${1#*=} # Delete everything up to "=" and assign the remainder.
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
# Backup old tutorials and removing tutorials directory
# --------------------------------------------------------------------------------------------------------------------------------

backup_old_tutorials(){
    local dest_dir="$1"
    local tutorials_dir="$2"
    if [ -z "${dry_run}" ]; then
        dt=$(date '+%Y%m%d%H%M%S');
        old_tutorials_dir="${dest_dir}/tutorials.old/${dt}"
        echo "Moving existing '${tutorials_dir}' to ${old_tutorials_dir}'..."
        mkdir -p "${old_tutorials_dir}"
        cp -rf "${tutorials_dir}/." "${old_tutorials_dir}" || echo "$tutorials_dir is missing, skipping backup"
        rm -rf "${tutorials_dir}"
        mkdir -p "${tutorials_dir}"
    fi
    }

# --------------------------------------------------------------------------------------------------------------------------------
# Function to verify align_tutorials is present, otherwise grab it
# --------------------------------------------------------------------------------------------------------------------------------

verify_align_tutorials(){
    local tutorials_dir="$1"
    local branch="$2"
    if [[ ! -f "${tutorials_dir}/align_tutorials.sh" || ${branch}<"v1.7" ]]; then
        echo "align_tutorials.sh is missing or old, downloading .."
        wget -O "${tutorials_dir}/align_tutorials.sh" https://raw.githubusercontent.com/mlrun/mlrun/development/docs/tutorials/align_tutorials.sh
    fi
    }

# --------------------------------------------------------------------------------------------------------------------------------
# Printing flags - dry_run and no_backup
# --------------------------------------------------------------------------------------------------------------------------------

# Don't download new tutorials only print them
# shellcheck disable=SC2236
if [ ! -z "${dry_run}" ]; then
    echo "Dry run; no files will be copied."
fi
# Don't back up old tutorials
# shellcheck disable=SC2236
if [ ! -z "${no_backup}" ]; then
    echo "The existing tutorials directory won't be backed up before the update."
fi

# --------------------------------------------------------------------------------------------------------------------------------
# Detecting tutorials_dir and dest_dir
# if --path is provided (tutorials_dir):
#    use it to create destination directory (parent directory)
# else:
#    if --user or V3IO_USERNAME is provided:
#        use it to create tutorials_dir and dest_dir
#    else: (no user and no v3io_username)
#        use pwd/tutorials as tutorials_dir and pwd as dest_dir
# --------------------------------------------------------------------------------------------------------------------------------

current_dir=$(pwd)
folder_name=$(basename "$PWD")
parent_dir=$(dirname "$(pwd)")
# cd to avoid running shell from deleted path
cd
if [ "${tutorials_dir}" ]; then # means --path is specified
    dest_dir=${tutorials_dir%/*} # taking parent dir
fi

# Case username isn't provided via command and `V3IO_USERNAME` env variable isn't declared
if [[ -z "${user}" && -z "${tutorials_dir}" ]]; then
    echo "--user and --path argument are empty, using local path"
    # To support when running from inside tutorials folder. for arbitrary folder add - /tutorials
    if [[ "${folder_name}" == "tutorials" ]]; then
        tutorials_dir="${current_dir}"
        dest_dir="${parent_dir}"
    else
        tutorials_dir="${current_dir}/tutorials"
        dest_dir="${current_dir}"
    fi
fi

# when --path isn't specified and either V3IO_USERNAME or --user is specified (otherwise case caught above).
if [ -z "${tutorials_dir}" ]; then
    dest_dir="/v3io/users/${user}"
    tutorials_dir="${dest_dir}/tutorials"
fi

# --------------------------------------------------------------------------------------------------------------------------------
# Printing arguments
# --------------------------------------------------------------------------------------------------------------------------------

echo "Updating tutorials ..
Username : $user
Tutorials directory : $tutorials_dir"
if [ -z $branch ]; then
    echo "Branch isn't specified, using mlrun version."
else
    echo "branch : $branch"
fi
if [ -z $mlrun_version ]; then
    echo "mlrun version isn't specified, aligning with installed mlrun version."
else
    echo "mlrun version : $mlrun_version"
fi
# IF BOTH branch and mlrun_ver are specified, raise an error to select only one !
if [[ -n "$branch" && -n "$mlrun_version" ]]; then
    error_usage "please specify only one, branch or mlrun-ver."
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
        # Otherwise, add it to the list without "rc"
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
      # case mlrun doesnt have an rc (its a release) and tutorials doesn't have matching release (fetching latest rc)
      if [ -z "$formatted_rc" ]; then # rc is ""
        echo "${with_rc[*]}" | tr ' ' '\n' | sort -Vr | head -n 1
        return
      fi
      # case mlrun does have an rc - return its matching tutorials rc
      for item in "${all_rcs[@]}"; do
        if [[ $item == *"$formatted_rc"* ]]; then
          echo "$item"
          return
        fi
      done
      # coldn't find matching rc (mlrun does have an rc but tutorials doesn't have a matching one) returns latest rc
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
    wget -c "${tar_url}" -O mlrun-tutorials.tar
    tar -xf mlrun-tutorials.tar -C "${temp_dir}" --strip-components 1
    rm -rf mlrun-tutorials.tar
    }

# --------------------------------------------------------------------------------------------------------------------------------
# Main script.
# backup old tutorials.
# if --branch is specified, use it to fetch mlrun/mlrun repository.
# --------------------------------------------------------------------------------------------------------------------------------

mkdir -p "${tutorials_dir}" # in case the tutorials dir isn't exist
# Backup tutorials if needed and deleting tutorials directory
if [ -z "${no_backup}" ]; then
    backup_old_tutorials "$dest_dir" "$tutorials_dir"
else
    # if --dry-run flag isn't set, remove current tutorials
    if [ -z "${dry_run}" ]; then
        rm -rf "${tutorials_dir}"
        mkdir -p "${tutorials_dir}"
    fi
fi
# Downloading tar_url to temp dir
temp_dir=$(mktemp -d /tmp/temp-get-tutorials.XXXXXXXXXX)
# If branch is specified, download mlrun repository and derive tutorials folder to new temp directory
if [ "$branch" ]; then
    echo "using specified branch $branch"
    tar_url="${git_base_url}/archive/${branch}.tar.gz"
    download_tar_to_temp_dir "$tar_url" "$temp_dir"
    # make sure only tutorials content in mlrun/mlrun/docs left.
    new_temp_dir=$(mktemp -d /tmp/temp-get-tutorials.XXXXXXXXXX)
    # If this isn't a dry run, derive tutorials and copy them to tutorials_dir
    if [ -z "${dry_run}" ]; then
        cp -rf "${temp_dir}/docs/tutorials/." "${new_temp_dir}/tutorials"
        temp_dir="${new_temp_dir}/tutorials"
        verify_align_tutorials "${temp_dir}" "${branch}"
        cp -rf "${temp_dir}/." "$tutorials_dir"
    else
        echo "dry run, not copying from branch ${branch}"
        echo "Identified the following files to copy to '${dest_dir}':"
        cd ${temp_dir}
        cd docs/tutorials
        echo "$(ls -a)"
    fi
    # Finish here the script when --branch is specified
    exit
fi

# --------------------------------------------------------------------------------------------------------------------------------
# If branch isn't specified, trying to use --mlrun-ver or detect installed mlrun version
# When mlrun isn't installed, using 1.7.0
# --------------------------------------------------------------------------------------------------------------------------------

if [ -z "$mlrun_version" ]; then # Branch and mlrun version isn't specified. using installed mlrun version
    pip_mlrun=$(pip show mlrun | grep Version) || :
    if [ -z "${pip_mlrun}" ]; then
        mlrun_version="1.7.0"
        # error_exit "MLRun version not found. Aborting..."
        # no --branch and --mlrun-ver and mlrun isn't installed, using mlrun==1.7.0
    else
        echo "Detected MLRun version: ${pip_mlrun}"
        mlrun_version="${pip_mlrun##Version: }"
    fi
fi
echo "Looking for tutorials with MLRun version - ${mlrun_version}."
latest_tag=$(get_latest_tag "${mlrun_version}" "${git_owner}" "${git_repo}" "${git_base_url}" "${git_url}")
echo "release tag or latest tag : ${latest_tag}"
if [ -z "${latest_tag}" ]; then
     error_exit "Couldn't locate a Git tag with prefix 'v${mlrun_version}.*'."
fi
branch=${latest_tag#refs/tags/}
# tutorials are introduced to mlrun in 1.4.0
if [[ "${branch}">"v1.4" ]]; then
    tar_url="${git_base_url}/releases/download/${branch}/mlrun-tutorials.tar"
    if [[ "${branch}"<"v1.5" ]]; then # Folder name changed in 1.5.0
        folder_name="tutorial"
    else
        folder_name="tutorials"
    fi
else
    error_exit "mlrun must be >= 1.4"
fi
echo "Using tar_url ${tar_url} branch: ${branch}"
download_tar_to_temp_dir "$tar_url" "$temp_dir"
verify_align_tutorials "${temp_dir}/${folder_name}" "${branch}"
if [ -z "${dry_run}" ]; then
    echo "copy files from ${temp_dir}/tutorials to ${tutorials_dir}"
    cp -rf "$temp_dir/$folder_name/." "$tutorials_dir"
else
    echo "Identified the following files to copy to '${dest_dir}':"
    cd ${temp_dir}
    cd ${folder_name}
    echo "$(ls -a)"
fi
