(llm-prompt-artifcta)=
# LLM prompt artifacts

LLM prompt artifacts are defined by their prompt template, the model, and the generation configuration.

**In this section**
- [SDK](#sdk)
- [Prompt artifacts](#prompt-artifacts)
- [Logging LLM prompt artifacts](#logging-llm-prompt-artifacts)
- [Deleting prompt artifacts](#deleting-prompt-artifacts-using-the-sdk)
- [Viewing LLM-prompt artifacts using the SDK](#viewing-llm-prompt-artifacts-using-the-sdk)
- [Viewing LLM-prompt artifacts in the UI](#viewing-llm-prompt-artifacts-in-the-ui)

## SDK
- {py:class}`~mlrun.projects.MlrunProject.log_llm_prompt`: Logs an LLM prompt artifact to the current project.
- {py:class}`~mlrun.projects.MlrunProject.list_llm_prompts`: Lists LLM prompt artifacts in the current project with support for filtering.
- {py:class}`~mlrun.projects.MlrunProject.paginated_list_llm_prompts`: Retrieves a paginated list of LLM prompt artifacts in the current project.


## Deleting prompt artifacts using the SDK

Delete prompt artifacts with {py:class}`~mlrun.projects.MlrunProject.delete_artifact`.

Guidelines
- You cannot delete an LLM prompt artifact if there is a MEP attached to it.
- You cannot delete a model if there is an LLM prompt pointing at it (whether or not this LLM prompt has a model endpoint). 

## Viewing LLM-prompt artifacts using the SDK 
View the list of LLM prompt artifacts in the current project with {py:class}`~mlrun.projects.MlrunProject.list_llm_prompts` or {py:class}`~mlrun.projects.MlrunProject.paginated_list_llm_prompts` for a paginated list. When using {py:class}`~mlrun.projects.MlrunProject.list_llm_prompts` there are multiple options for filtering.

## Viewing LLM-prompt artifacts in the UI

The LLM prompts page lists all the prompt artifacts in the project. You can filter by lable, LLM prompt version tag, model name, and model version tag.

Each prompt template has these tabs, providing further details:
- Overview: 
  - General: Key, Description, Model name, Hash, Version tag, Original source, Iteration, URI, Path (if there is a path, the prompt template text is read from this path), UID,  Updated, Label
  - Producer: Name, Kind, Tag, Owner, UID
- Prompt template: 
  - Prompt: Searchable text displaying the roles and their content. entire prompt template with the placeholders (the {argument name}). You can minimize roles to get a better view of the other role(s).
  - Arguments: Lists the  arguments and their descriptions.
- Generation configuration: Displays the keys and their values, or indicates that the prompt template uses the default configuration.



