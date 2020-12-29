# k8s label value format
# https://github.com/kubernetes/kubernetes/blob/master/staging/src/k8s.io/apimachinery/pkg/util/validation/validation.go#L161
label_value = [r"^.{0,63}$", r"^(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?$"]

# DNS Subdomain (RFC 1123) - used by k8s for most resource names format
# https://github.com/kubernetes/kubernetes/blob/master/staging/src/k8s.io/apimachinery/pkg/util/validation/validation.go#L204
dns_1123_subdomain = [
    r"^.{0,253}$",
    r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*$",
]

# DNS Label (RFC 1123) - used by k8s for resource names format
# https://github.com/kubernetes/kubernetes/blob/master/staging/src/k8s.io/apimachinery/pkg/util/validation/validation.go#L183
dns_1123_label = [
    r"^.{0,63}$",
    r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$",
]

run_name = label_value

# A project name have the following restrictions:
# It should be a valid Nuclio Project CRD name which is dns 1123 subdomain
# It should be a valid k8s label value since Nuclio use the project name in labels of resources
# It should be a valid namespace name (cause we plan to map it to one) which is dns 1123 label
# of the 3 restrictions, dns 1123 label is the most strict, so we enforce only it
project_name = dns_1123_label
