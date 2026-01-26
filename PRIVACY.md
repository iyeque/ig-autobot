Privacy Policy
Version: 1.0
Last updated: 2026-01-26

## Introduction
This Privacy Policy describes what information the ig-autobot project may collect, how that information is used, how it may be shared, and the choices available to people who interact with the project. This policy applies to the code, GitHub Actions workflows, and any hosted endpoints or services used by the repository.

## Data Collected
Automatically collected data
- Workflow and runtime logs (timestamps, run IDs, HTTP status codes, error messages).
- Metadata produced by integrations and hosting services.
User provided data
- Configuration values and secrets you supply to the project (API tokens, model slugs, account IDs).
- Content you add to the content queue (post prompts, images, captions).
Generated content
- Images, captions, and other artifacts produced by AI services during bot runs.
Third party data
- Responses and metadata returned by external APIs and providers used by the project.

## How Data Is Used
Operation
- To select, generate, host, and publish social posts as configured by the repository owner.
Maintenance and debugging
- To diagnose failures, improve reliability, and maintain state (for example, which posts have been used).
Analytics
- Optional collection of non‑personal metrics (success/failure rates, latency) to improve the system.
No advertising profiling
- The project does not use collected data to build advertising profiles or sell personal data.

## Sharing and Disclosure
Service providers
- Data may be shared with third‑party providers strictly to perform the service (AI inference providers, image hosts, social platform APIs). These providers act as processors under your direction.
Legal requirements
- Data may be disclosed to comply with legal obligations or to protect rights and safety.
No sale of personal data
- The project does not sell personal data to third parties.

## Secrets and Credentials
Storage
- Secrets (API tokens, keys) must be stored in the repository’s secret management (CI secrets) and must never be committed to source.
Access
- Only authorized CI workflows and maintainers should have access to secrets. Rotate and revoke tokens regularly.
Best practice
- Use least‑privilege tokens and scoped credentials. Prefer Page tokens for publishing and scoped inference tokens for AI providers.

## Data Retention and Deletion
Retention
- Generated content and logs are retained according to repository and workflow configuration.
Deletion
- To remove content or logs, delete the files from the repository and purge any hosted copies. Revoke or rotate any tokens you no longer want active.
Formal deletion requests
- If a formal deletion procedure is required, document the steps in the repository’s operational notes and follow them.

## Security
Measures
- Use encrypted secrets, HTTPS for external calls, and minimal permissions for tokens. Keep dependencies up to date and monitor workflow logs for suspicious activity.
Limitations
- No system is perfectly secure. Follow best practices for credential management and incident response.

## Your Rights and Choices
Access and correction
- If personal data is stored by the project, you may request access or correction from the repository owner.
Opt out
- To stop automated posting for a given account, remove or rotate the publishing token and disable the scheduled workflow.
Data portability
- Generated content and logs stored in the repository can be exported by cloning the repository.

## Third Party Services
External providers
- The project uses third‑party AI and social platform APIs. Their privacy practices apply when you use those services. Review provider privacy policies before enabling integrations.
Hosting
- If images or other artifacts are hosted externally, those hosts may collect access logs and metadata.

## Children
Age restriction
- The project is not intended for use by children under applicable legal ages. Do not provide personal data of minors to the project.

## Changes to This Policy
Updates
- This policy may be updated. The repository owner should record the policy version and last updated date in the published policy. Significant changes should be communicated to collaborators.

## Contact
Questions and requests
- Direct privacy questions, data access, or deletion requests to the repository owner or the contact method listed in the repository.
