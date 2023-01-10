get_os() {
  unameOut="$(uname -s)"
  case "${unameOut}" in
      Linux*)     os=Linux;;
      Darwin*)    os=Mac;;
      *)          os="UNKNOWN:${unameOut}"
  esac
  echo ${os}
}

SCHEMAS_DIR=../mlrun/api/proto/
SED_REGEX='s/mlrun.api.proto/./g'
OS=$(get_os)
SCHEMA_FILES=$(find ../mlrun/api/proto/ -name '*pb2*.py')

if [ "${OS}" == "Mac" ]; then
  sed -i '' -e ${SED_REGEX} ${SCHEMA_FILES}
else
  sed -i -e ${SED_REGEX} ${SCHEMA_FILES}
fi
