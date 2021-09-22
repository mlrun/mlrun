# k8s label value format
# https://github.com/kubernetes/kubernetes/blob/v1.20.0/staging/src/k8s.io/apimachinery/pkg/util/validation/validation.go#L161
label_value = [r"^.{0,63}$", r"^(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?$"]

# DNS Subdomain (RFC 1123) - used by k8s for most resource names format
# https://github.com/kubernetes/kubernetes/blob/v1.20.0/staging/src/k8s.io/apimachinery/pkg/util/validation/validation.go#L204
dns_1123_subdomain = [
    r"^.{0,253}$",
    r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*$",
]

# DNS Label (RFC 1123) - used by k8s for resource names format
# https://github.com/kubernetes/kubernetes/blob/v1.20.0/staging/src/k8s.io/apimachinery/pkg/util/validation/validation.go#L183
dns_1123_label = [
    r"^.{0,63}$",
    r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$",
]

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
#   generated_resource_name - for each resource added a describing suffix (e.g driver) in that case
#       the longest suffix is "-[uid*13]-driver-svc" which contains 25 characters. Therefore the limit should be
#       63 - 25 - 9 = 29
#       NOTE: If a name is between 30-33 characters - the function will complete successfully without creating the
#           driver-svc meaning there is no way to get the response through a ui
sparkjob_name = label_value + [r"^.{0,29}$"]

# A project name have the following restrictions:
# It should be a valid Nuclio Project CRD name which is dns 1123 subdomain
# It should be a valid k8s label value since Nuclio use the project name in labels of resources
# It should be a valid namespace name (cause we plan to map it to one) which is dns 1123 label
# of the 3 restrictions, dns 1123 label is the most strict, so we enforce only it
project_name = dns_1123_label

secret_key = k8s_secret_and_config_map_key
