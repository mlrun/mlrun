# Copyright 2023 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# pipeline param format which is passed when running a pipeline (e.g. {{pipelineparam:op=;name=mem}})
# https://github.com/kubeflow/pipelines/blob/16edebf4eaf84cd7478e2601ef4878ab339a7854/sdk/python/kfp/dsl/_pipeline_param.py#L213
# this is expected to be resolved at runtime
pipeline_param = [r"{{pipelineparam:op=([\w\s_-]*);name=([\w\s_-]+)}}"]

# k8s character limit is for 63 characters
k8s_character_limit = [r"^.{0,63}$"]

# k8s name
# https://github.com/kubernetes/apimachinery/blob/kubernetes-1.25.16/pkg/util/validation/validation.go#L33
qualified_name = [r"^(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?$"]

# k8s label value format
# https://github.com/kubernetes/kubernetes/blob/v1.20.0/staging/src/k8s.io/apimachinery/pkg/util/validation/validation.go#L161
label_value = k8s_character_limit + qualified_name

# DNS Subdomain (RFC 1123) - used by k8s for most resource names format
# https://github.com/kubernetes/kubernetes/blob/v1.20.0/staging/src/k8s.io/apimachinery/pkg/util/validation/validation.go#L204
dns_1123_subdomain = [
    r"^.{0,253}$",
    r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*$",
]

# DNS Label (RFC 1123) - used by k8s for resource names format
# https://github.com/kubernetes/kubernetes/blob/v1.20.0/staging/src/k8s.io/apimachinery/pkg/util/validation/validation.go#L183
dns_1123_label = k8s_character_limit + [r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$"]

# DNS 1035 - used by k8s for services services
# https://github.com/kubernetes/kubernetes/blob/v1.20.0/staging/src/k8s.io/apimachinery/pkg/util/validation/validation.go#L220
dns_1035_label = [r"[a-z]([-a-z0-9]*[a-z0-9])?"]

# https://github.com/kubernetes/kubernetes/blob/v1.20.0/staging/src/k8s.io/apimachinery/pkg/util/validation/validation.go#L424
k8s_secret_and_config_map_key = [
    r"^.{0,253}$",
    r"^[-._a-zA-Z0-9]+$",
]

# https://github.com/kubernetes/kubernetes/blob/v1.20.0/staging/src/k8s.io/apimachinery/pkg/api/resource/quantity.go#L136
k8s_resource_quantity_regex = [r"^([+-]?[0-9.]+)([eEinumkKMGTP]*[-+]?[0-9]*)$"]

run_name = label_value

# Sparkjob name value format
# The actual limit is 63 characters, but due to mlrun and spark operator additions for unique
# values - the limit is set to 30.
# The names of the generated resources are in the format: [function_name]-[uid*8]-[generated_resource_name]
#   function_name - the name provided by the user
#   uid*8 - is 8 characters generated in mlrun to give the resources a guaranteed unique name.
#   generated_resource_name - for each resource added a describing suffix (e.g. driver) in that case
#       the longest suffix is "-[uid*13]-driver-svc" which contains 25 characters. Therefore the limit should be
#       63 - 25 - 9 = 29
#       NOTE: If a name is between 30-33 characters - the function will complete successfully without creating the
#           driver-svc meaning there is no way to get the response through a ui
sprakjob_length = [r"^.{0,29}$"]

# part of creating sparkjob operator it creates a service with the same name of the job
sparkjob_service_name = dns_1035_label

sparkjob_name = label_value + sprakjob_length + sparkjob_service_name

# A project name have the following restrictions:
# It should be a valid Nuclio Project CRD name which is dns 1123 subdomain
# It should be a valid k8s label value since Nuclio use the project name in labels of resources
# It should be a valid namespace name (because we plan to map it to one) which is dns 1123 label
# of the 3 restrictions, dns 1123 label is strictest, so we enforce only it
project_name = dns_1123_label

# Special characters are not permitted in tag names because they can be included in the url and cause problems.
# We only accept letters, capital letters, numbers, dots, and hyphens, with a k8s character limit.
tag_name = label_value

secret_key = k8s_secret_and_config_map_key

artifact_key = [r"^[A-Za-z0-9]([-A-Za-z0-9_.]*[A-Za-z0-9])?$"]

# must not start with _
# must be alphanumeric or _
# max 256 length
v3io_stream_consumer_group = [r"^(?!_)[a-zA-Z0-9_]{1,256}$"]

# URI patterns
run_uri_pattern = r"^(?P<project>.*)@(?P<uid>.*)\#(?P<iteration>.*?)(:(?P<tag>.*))?$"

artifact_uri_pattern = (
    r"^((?P<project>.*)/)?"  # Optional project
    r"(?P<key>.*?)"  # Key
    r"(\#(?P<iteration>.*?))?"  # Optional iteration
    r"(:(?P<tag>.*?))?"  # Optional tag
    r"(@(?P<tree>.*?))?"  # Optional tree
    r"(\^(?P<uid>.*))?$"  # Optional uid
)

artifact_producer_uri_pattern = (
    r"^((?P<project>.*)/)?(?P<uid>.*?)(\-(?P<iteration>.*?))?$"
)

mail_regex = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
alert_name_regex = r"^[a-zA-Z0-9-_]+$"
