import dagger
from dagger import dag, function, object_type, Doc, DefaultPath
from typing import Annotated


@object_type
class DatasetEc2LoadBenchmarks:
    @function
    def container_echo(self, string_arg: str) -> dagger.Container:
        """Returns a container that echoes whatever string argument is provided"""
        return dag.container().from_("alpine:latest").with_exec(["echo", string_arg])

    @function
    async def grep_dir(self, directory_arg: dagger.Directory, pattern: str) -> str:
        """Returns lines that match a pattern in the files of the provided Directory"""
        return await (
            dag.container()
            .from_("alpine:latest")
            .with_mounted_directory("/mnt", directory_arg)
            .with_workdir("/mnt")
            .with_exec(["grep", "-R", pattern, "."])
            .stdout()
        )

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

         AWS_DEFAULT_REGION="eu-central-1" sudo -E dagger -c '. --region env://AWS_DEFAULT_REGION --access-key-id env://AWS_ACCESS_KEY_ID --secret-access-key env://AWS_SECRET_ACCESS_KEY --aws-account data | ec2-deploy'

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
                        f'terraform -chdir=infrastructure/ec2 apply -var-file="../tfvars/{self.aws_account}.tfvars.json" -var-file="../tfvars/ec2.tfvars.json" -auto-approve -no-color'
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

         AWS_DEFAULT_REGION="eu-central-1" sudo -E dagger -c '. --region env://AWS_DEFAULT_REGION --access-key-id env://AWS_ACCESS_KEY_ID --secret-access-key env://AWS_SECRET_ACCESS_KEY --aws-account data | ec2-destroy'

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
                        f'terraform -chdir=infrastructure/ec2 destroy -var-file="../tfvars/{self.aws_account}.tfvars.json" -var-file="../tfvars/ec2.tfvars.json" -auto-approve -no-color'
                    ),
                ]
            )
            .stdout()
        )
