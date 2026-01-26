import dagger
from dagger import dag, function, object_type, Doc, DefaultPath
from typing import Annotated


@object_type
class DatasetEc2LoadBenchmarks:
    region: Annotated[dagger.Secret | None, Doc("AWS_DEFAULT_REGION")] = None
    access_key_id: Annotated[dagger.Secret | None, Doc("ID AWS_ACCESS_KEY_ID")] = None
    secret_access_key: Annotated[
        dagger.Secret | None, Doc("Secret AWS_SECRET_ACCESS_KEY")
    ] = None
    aws_account: Annotated[
        str,
        Doc(
            "The AWS account where the keys are coming from (data, staging, production)"
        ),
    ] = "staging"

    # TODO: This data should be extracted from the infra json files in the future
    infra_project_name: Annotated[str, Doc("Name of infra project")] = (
        "dataset-load-benchmarks"
    )

    @function
    def train_ssh_command(
        self,
        source: Annotated[
            dagger.Directory,
            DefaultPath("/"),
            Doc("location of directory containing ssh key"),
        ],
        ec2_dns: Annotated[
            str,
            Doc("location of directory containing Dockerfile"),
        ],
    ) -> str:
        """
        Mount required infra for the lambda service to work

        NOTE: This can be run with a command like:

         AWS_DEFAULT_REGION="us-east-1" sudo -E dagger -c '. --region env://AWS_DEFAULT_REGION --access-key-id env://AWS_ACCESS_KEY_ID --secret-access-key env://AWS_SECRET_ACCESS_KEY --aws-account data | train-ssh-command $HOME'

        having exported the AWS secrets first
        """

        return (
            dag.container()
            .from_("debian:bookworm-slim")
            .with_secret_variable("AWS_DEFAULT_REGION", self.region)
            .with_secret_variable("AWS_ACCESS_KEY_ID", self.access_key_id)
            .with_secret_variable("AWS_SECRET_ACCESS_KEY", self.secret_access_key)
            .with_directory("/src", source)
            .with_workdir("/src")
            .with_exec(
                [
                    "sh",
                    "-c",
                    (
                        f"ssh -i .ssh/wirisml-sharedkey.pem ubuntu@{ec2_dns} -t 'git clone ; bash -l'"
                    ),
                ]
            )
            .stdout()
        )

    @function
    def init_tf_states_backend(
        self,
        source: Annotated[
            dagger.Directory,
            DefaultPath("/"),
            Doc("location of directory containing Dockerfile"),
        ],
    ) -> str:
        """
        Mount required infra for the lambda service to work

        NOTE: This can be run with a command like:

         AWS_DEFAULT_REGION="us-east-1" sudo -E dagger -c '. --region env://AWS_DEFAULT_REGION --access-key-id env://AWS_ACCESS_KEY_ID --secret-access-key env://AWS_SECRET_ACCESS_KEY --aws-account data | init-tf-states-backend'

        having exported the AWS secrets first
        """

        bucket_name = (
            f"wiris-{self.infra_project_name}-{self.aws_account}-terraform-states"
        )

        return (
            dag.container()
            .from_("amazon/aws-cli")
            .with_secret_variable("AWS_DEFAULT_REGION", self.region)
            .with_secret_variable("AWS_ACCESS_KEY_ID", self.access_key_id)
            .with_secret_variable("AWS_SECRET_ACCESS_KEY", self.secret_access_key)
            .with_directory("/src", source)
            .with_workdir("/src")
            .with_exec(
                [
                    "sh",
                    "-c",
                    (
                        f"aws s3api create-bucket --bucket {bucket_name} --region $AWS_DEFAULT_REGION"
                    ),
                ]
            )
            .with_exec(
                [
                    "sh",
                    "-c",
                    (
                        f'echo -e "Sample code for the \\"backend.{self.aws_account}.conf\\" in any infra (<INFRA_NAME>)\n"'
                        '"---\n"'
                        f'"bucket = \\"{bucket_name}\\"\n"'
                        '"key = \\"<INFRA_NAME>.state\\""'
                    ),
                ]
            )
            .stdout()
        )

    # TODO: Add an initial function for the required key pair creation in AWS for the instance access
    # @function
    # def init_key_pair(

    @function
    def ec2_deploy(
        self,
        source: Annotated[
            dagger.Directory,
            DefaultPath("/"),
            Doc("location of directory containing Dockerfile"),
        ],
    ) -> str:
        """
        Mount required infra for the lambda service to work

        NOTE: This can be run with a command like:

         AWS_DEFAULT_REGION="us-east-1" sudo -E dagger -c '. --region env://AWS_DEFAULT_REGION --access-key-id env://AWS_ACCESS_KEY_ID --secret-access-key env://AWS_SECRET_ACCESS_KEY --aws-account data | ec-2-deploy'

        having exported the AWS secrets first
        """

        return (
            dag.container()
            .from_("hashicorp/terraform")
            .with_secret_variable("AWS_DEFAULT_REGION", self.region)
            .with_secret_variable("AWS_ACCESS_KEY_ID", self.access_key_id)
            .with_secret_variable("AWS_SECRET_ACCESS_KEY", self.secret_access_key)
            .with_directory("/src", source)
            .with_workdir("/src")
            .with_exec(
                [
                    "sh",
                    "-c",
                    (
                        f'terraform -chdir=infrastructure/ec2 init -backend-config="backend.{self.aws_account}.conf"'
                    ),
                ]
            )
            .with_exec(
                [
                    "sh",
                    "-c",
                    (
                        # f'terraform -chdir=infrastructure/ec2 apply -var-file="../tfvars/{self.aws_account}.tfvars.json" -var-file="../tfvars/ec2.tfvars.json" -auto-approve -no-color'
                        "terraform -chdir=infrastructure/ec2 apply -auto-approve -no-color"
                    ),
                ]
            )
            .stdout()
        )

    @function
    def ec2_destroy(
        self,
        source: Annotated[
            dagger.Directory,
            DefaultPath("/"),
            Doc("location of directory containing Dockerfile"),
        ],
    ) -> str:
        """
        Mount required infra for the lambda service to work

        NOTE: This can be run with a command like:

         AWS_DEFAULT_REGION="us-east-1" sudo -E dagger -c '. --region env://AWS_DEFAULT_REGION --access-key-id env://AWS_ACCESS_KEY_ID --secret-access-key env://AWS_SECRET_ACCESS_KEY --aws-account data | ec-2-destroy'

        having exported the AWS secrets first
        """

        return (
            dag.container()
            .from_("hashicorp/terraform")
            .with_secret_variable("AWS_DEFAULT_REGION", self.region)
            .with_secret_variable("AWS_ACCESS_KEY_ID", self.access_key_id)
            .with_secret_variable("AWS_SECRET_ACCESS_KEY", self.secret_access_key)
            .with_directory("/src", source)
            .with_workdir("/src")
            .with_exec(
                [
                    "sh",
                    "-c",
                    (
                        f'terraform -chdir=infrastructure/ec2 init -backend-config="backend.{self.aws_account}.conf"'
                    ),
                ]
            )
            .with_exec(
                [
                    "sh",
                    "-c",
                    (
                        # f'terraform -chdir=infrastructure/ec2 destroy -var-file="../tfvars/{self.aws_account}.tfvars.json" -var-file="../tfvars/ec2.tfvars.json" -auto-approve -no-color'
                        "terraform -chdir=infrastructure/ec2 destroy -auto-approve -no-color"
                    ),
                ]
            )
            .stdout()
        )
