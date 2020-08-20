# k8s label value format
label_value = [r"^.{0,63}$", r"^(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?$"]

# DNS label (RFC 1123) - used by k8s for most resource names format
dns_1123_label = [r"^.{0,63}$", r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$"]

run_name = label_value
