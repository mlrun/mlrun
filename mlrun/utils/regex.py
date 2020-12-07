# k8s label value format
# https://github.com/kubernetes/kubernetes/blob/master/staging/src/k8s.io/apimachinery/pkg/util/validation/validation.go#L161
label_value = [r"^.{0,63}$", r"^(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?$"]

# DNS Subdomain (RFC 1123) - used by k8s for most resource names format
# https://github.com/kubernetes/kubernetes/blob/master/staging/src/k8s.io/apimachinery/pkg/util/validation/validation.go#L204
dns_1123_subdomain = [
    r"^.{0,253}$",
    r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*$",
]

run_name = label_value

project_name = dns_1123_subdomain
