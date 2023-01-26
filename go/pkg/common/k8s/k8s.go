package k8s

import "os"

func IsInKubernetesCluster() bool {
	return len(os.Getenv("KUBERNETES_SERVICE_HOST")) != 0 && len(os.Getenv("KUBERNETES_SERVICE_PORT")) != 0
}

func ResolveRunningNamespace(namespaceArgument string) string {

	// if the namespace was passed in the arguments, use that
	if namespaceArgument != "" {
		return namespaceArgument
	}

	// if the namespace exists in env, use that
	if namespaceEnv := os.Getenv("MLRUN_LOG_COLLECTOR__NAMESPACE"); namespaceEnv != "" {
		return namespaceEnv
	}

	// try reading from file if running in k8s
	if IsInKubernetesCluster() {
		if namespacePod, err := os.ReadFile("/var/run/secrets/kubernetes.io/serviceaccount/namespace"); err == nil {
			return string(namespacePod)
		}
	}

	return "default"
}
