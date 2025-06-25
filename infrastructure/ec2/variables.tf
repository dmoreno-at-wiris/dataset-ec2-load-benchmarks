variable "project" {
  description = "Project name"
  type        = string
  default     = "emilies"
}

variable "environment" {
  description = "Environment name (staging and production)"
  type        = string
  default     = "staging"
}

# variable "lambdas" {
#   description = "Defines information about the lambdas"
#   type        = map(any)
#   default = {
#     foobar = {
#       path                           = "abc"
#       docker_tag                     = "abc"
#       memory_size                    = 2048
#       timeout                        = 10
#       stable_version_number          = 0
#       previous_stable_version_number = 0
#     }
#   }
# }
