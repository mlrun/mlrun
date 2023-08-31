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

import base64
import json
import subprocess

import click


def run_click_command(command, **kwargs):
    """
    Runs a click command with the specified arguments.
    :param command: The click command to run.
    :param kwargs: Keyword arguments to pass to the click command.
    """
    # create a Click context object
    ctx = click.Context(command)
    # invoke the Click command with the desired arguments
    ctx.invoke(command, **kwargs)


def get_installed_releases(namespace):
    cmd = ["helm", "ls", "-n", namespace, "--deployed", "--short"]
    output = subprocess.check_output(cmd).decode("utf-8")
    release_names = output.strip().split("\n")
    return release_names


def run_command(cmd):
    """
    Runs a shell command and returns its output and exit status.
    """
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    return result.stdout.decode("utf-8"), result.returncode


def create_ingress_resource(domain_name, ipadd):
    # Replace the placeholder string with the actual domain name
    yaml_manifest = """
    apiVersion: networking.k8s.io/v1
    kind: Ingress
    metadata:
      annotations:
        nginx.ingress.kubernetes.io/auth-cache-duration: 200 202 5m, 401 30s
        nginx.ingress.kubernetes.io/auth-cache-key: $host$http_x_remote_user$http_cookie$http_authorization
        nginx.ingress.kubernetes.io/proxy-body-size: "0"
        nginx.ingress.kubernetes.io/whitelist-source-range: "{}"
        nginx.ingress.kubernetes.io/service-upstream: "true"
        nginx.ingress.kubernetes.io/ssl-redirect: "false"
      labels:
        release: redisinsight
      name: redisinsight
      namespace: devtools
    spec:
      ingressClassName: nginx
      rules:
      - host: {}
        http:
          paths:
          - backend:
              service:
                name: redisinsight
                port:
                  number: 80
            path: /
            pathType: ImplementationSpecific
      tls:
      - hosts:
        - {}
        secretName: ingress-tls
    """.format(
        ipadd, domain_name, domain_name
    )
    subprocess.run(
        ["kubectl", "apply", "-f", "-"], input=yaml_manifest.encode(), check=True
    )


def get_ingress_controller_version():
    # Run the kubectl command and capture its output
    kubectl_cmd = "kubectl"
    namespace = "default-tenant"
    grep_cmd = "grep shell.default-tenant"
    awk_cmd1 = "awk '{print $3}'"
    awk_cmd2 = "awk -F shell.default-tenant '{print $2}'"
    cmd = f"{kubectl_cmd} get ingress -n {namespace} | {grep_cmd} | {awk_cmd1} | {awk_cmd2}"
    result = subprocess.run(
        cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    return result.stdout.decode("utf-8").strip()


def get_svc_password(namespace, service_name, key):
    cmd = f'kubectl get secret --namespace {namespace} {service_name} -o jsonpath="{{.data.{key}}}" | base64 --decode'
    result = subprocess.run(
        cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    return result.stdout.decode("utf-8").strip()


def print_svc_info(svc_host, svc_port, svc_username, svc_password, nodeport):
    print(f"Service is running at {svc_host}:{svc_port}")
    print(f"Service username: {svc_username}")
    print(f"Service password: {svc_password}")
    print(f"service nodeport: {nodeport}")


def check_redis_installation():
    cmd = "helm ls -A | grep -w redis | awk '{print $1}' | wc -l"
    result = subprocess.check_output(cmd, shell=True)
    return result.decode("utf-8").strip()


def add_repos():
    repos = {"bitnami": "https://charts.bitnami.com/bitnami"}
    for repo, url in repos.items():
        cmd = f"helm repo add {repo} {url}"
        subprocess.run(cmd.split(), check=True)


def install_redisinsight(ipadd):
    print(check_redis_installation)
    if check_redis_installation() == "1":
        subprocess.run(["rm", "-rf", "redisinsight-chart-0.1.0.tgz*"])
        chart_url = "https://docs.redis.com/latest/pkgs/redisinsight-chart-0.1.0.tgz"
        chart_file = "redisinsight-chart-0.1.0.tgz"
        subprocess.run(["wget", chart_url])
        # get redis password
        redis_password = subprocess.check_output(
            [
                "kubectl",
                "get",
                "secret",
                "--namespace",
                "devtools",
                "redis",
                "-o",
                'jsonpath="{.data.redis-password}"',
            ],
            encoding="utf-8",
        ).strip('"\n')
        redis_password = base64.b64decode(redis_password).decode("utf-8")
        cmd = [
            "helm",
            "install",
            "redisinsight",
            chart_file,
            "--set",
            "redis.url=redis-master",
            "--set",
            "master.service.nodePort=6379",
            "--set",
            f"auth.password={redis_password}",
            "--set",
            "fullnameOverride=redisinsight",
            "--namespace",
            "devtools",
        ]
        subprocess.run(cmd, check=True)
        # run patch cmd
        fqdn = get_ingress_controller_version()
        full_domain = "redisinsight" + fqdn
        create_ingress_resource(full_domain, ipadd)
        deployment_name = "redisinsight"
        container_name = "redisinsight-chart"
        env_name = "RITRUSTEDORIGINS"
        full_domain = full_domain
        pfull_domain = "https://" + full_domain
        patch_command = (
            f'kubectl patch deployment -n devtools {deployment_name} -p \'{{"spec":{{"template":{{"spec":{{'
            f'"containers":[{{"name":"{container_name}","env":[{{"name":"{env_name}","value":"'
            f"{pfull_domain}\"}}]}}]}}}}}}}}'"
        )
        subprocess.run(patch_command, shell=True)
        clean_command = "rm -rf redisinsight-chart-0.1.0.tgz*"
        subprocess.run(clean_command, shell=True)
    else:
        print("redis is not install, please install redis first")
        exit()


@click.command()
@click.option("--redis", is_flag=True, help="Install Redis")
@click.option("--kafka", is_flag=True, help="Install Kafka")
@click.option("--mysql", is_flag=True, help="Install MySQL")
@click.option("--redisinsight", is_flag=True, help="Install Redis GUI")
@click.option("--ipadd", default="localhost", help="IP address as string")
def install(redis, kafka, mysql, redisinsight, ipadd):
    # Check if the local-path storage class exists
    output, exit_code = run_command(
        "kubectl get storageclass local-path >/dev/null 2>&1"
    )
    if exit_code != 0:
        # Install the local-path provisioner
        cmd = (
            "kubectl apply -f https://raw.githubusercontent.com/rancher/local-path-provisioner/v0.0.24/deploy/local"
            "-path-storage.yaml"
        )
        output, exit_code = run_command(cmd)
        if exit_code == 0:
            # Set the local-path storage class as the default
            cmd = (
                'kubectl patch storageclass local-path -p \'{"metadata": {"annotations":{'
                '"storageclass.kubernetes.io/is-default-class":"true"}}}\''
            )
            output, exit_code = run_command(cmd)
            if exit_code == 0:
                print(
                    "local-path storage class has been installed and set as the default."
                )
            else:
                print(f"Error setting local-path storage class as default: {output}")
        else:
            print(f"Error installing local-path storage class: {output}")
    else:
        print("local-path storage class already exists.")
    services = {
        "redis": {
            "chart": "bitnami/redis",
            "set_values": "--set master.service.nodePorts.redis=31001",
        },
        "kafka": {
            "chart": "bitnami/kafka",
            "set_values": "--set service.nodePorts.client=31002",
        },
        "mysql": {
            "chart": "bitnami/mysql",
            "set_values": "--set primary.service.nodePorts.mysql=31003",
        },
    }
    namespace = "devtools"
    # Add Helm repos
    add_repos()
    # Check if the namespace exists, if not create it
    check_namespace_cmd = f"kubectl get namespace {namespace}"
    try:
        subprocess.run(check_namespace_cmd.split(), check=True)
    except subprocess.CalledProcessError:
        create_namespace_cmd = f"kubectl create namespace {namespace}"
        subprocess.run(create_namespace_cmd.split(), check=True)
    for service, data in services.items():
        if locals().get(service):
            chart = data["chart"]
            set_values = data["set_values"]
            cmd = f"helm install {service} {chart} {set_values} --namespace {namespace}"
            print(cmd)
            subprocess.run(cmd.split(), check=True)
    if redisinsight:
        install_redisinsight(ipadd)


@click.command()
@click.option("--redis", is_flag=True, help="Uninstall Redis")
@click.option("--kafka", is_flag=True, help="Uninstall Kafka")
@click.option("--mysql", is_flag=True, help="Uninstall MySQL")
@click.option("--redisinsight", is_flag=True, help="Uninstall Redis GUI")
def uninstall(redis, kafka, mysql, redisinsight):
    services = ["redis", "kafka", "mysql", "redisinsight"]
    namespace = "devtools"
    try:
        if redisinsight:
            cmd = "kubectl delete ingress -n devtools redisinsight"
            subprocess.run(cmd.split(), check=True)
    except Exception as e:
        print(e)
    try:
        for service in services:
            if locals().get(service):
                cmd = f"helm uninstall {service} --namespace {namespace}"
                subprocess.run(cmd.split(), check=True)
    except Exception as e:
        print(e)
    try:
        print("namespace deleteted")
        cmd = "kubectl delete namespace  devtools"
        subprocess.run(cmd.split(), check=True)
    except Exception as e:  # !!!
        print(e)
        pass
        # code to handle any exception


@click.command()
def list_services():
    namespace = "devtools"
    # for service in services:
    cmd = f"helm ls  --namespace {namespace} "
    subprocess.run(cmd.split(), check=True)


def list_services_h():
    namespace = "devtools"
    return get_installed_releases(namespace)


@click.command()
@click.option("--redis", is_flag=True, help="Get Redis info")
@click.option("--kafka", is_flag=True, help="Get Kafka info")
@click.option("--mysql", is_flag=True, help="Get MySQL info")
@click.option("--redisinsight", is_flag=True, help="Get Redis GUI info")
@click.option("--output", default="human", type=click.Choice(["human", "json"]))
def status(redis, kafka, mysql, redisinsight, output):
    namespace = "devtools"
    get_all_output = {}
    if redis:
        svc_password = get_svc_password(namespace, "redis", "redis-password")
        get_all_output["redis"] = status_h("redis")
        if output == "human":
            print_svc_info(
                "redis-master-0.redis-headless.devtools.svc.cluster.local",
                6379,
                "default",
                svc_password,
                "-------",
            )
    if kafka:
        get_all_output["kafka"] = status_h("kafka")
        if output == "human":
            print_svc_info("kafka", 9092, "-------", "-------", "-------")
    if mysql:
        svc_password = get_svc_password(namespace, "mysql", "mysql-root-password")
        get_all_output["mysql"] = status_h("mysql")
        if output == "human":
            print_svc_info("mysql", 3306, "root", svc_password, "-------")
    if redisinsight:
        get_all_output["redisinsight"] = status_h("redisinsight")
        if output == "human":
            print_svc_info(
                "",
                " " + get_all_output["redisinsight"]["app_url"],
                "-------",
                "-------",
                "-------",
            )

    if output == "json":
        print(json.dumps(get_all_output))


def status_h(svc):
    namespace = "devtools"
    if svc == "redis":
        svc_password = get_svc_password(namespace, "redis", "redis-password")
        dict = {
            "app_url": "redis-master-0.redis-headless.devtools.svc.cluster.local:6379",
            "username": "default",
            "password": svc_password,
        }
        return dict
    if svc == "kafka":
        dict = {"app_url": "kafka-0.kafka-headless.devtools.svc.cluster.local:9092"}
        return dict
    if svc == "mysql":
        svc_password = get_svc_password(namespace, "mysql", "mysql-root-password")
        dict = {
            "app_url": "mysql-0.mysql.devtools.svc.cluster.local:3306",
            "username": "root",
            "password": svc_password,
        }
        return dict
    if svc == "redisinsight":
        fqdn = get_ingress_controller_version()
        full_domain = "https://redisinsight" + fqdn
        dict = {"app_url": full_domain}
        return dict


@click.group()
def cli():
    pass


cli.add_command(install)
cli.add_command(uninstall)
cli.add_command(list_services)
cli.add_command(status)

if __name__ == "__main__":
    cli()
