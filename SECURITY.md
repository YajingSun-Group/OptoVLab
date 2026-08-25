# Security Policy

OptoVLab is a research prototype and does not currently provide authentication
or multi-tenant isolation. Run the full FastAPI and agent workbench only on a
trusted network or behind an authenticated reverse proxy.

Never expose model API keys, unpublished PDFs, runtime databases, scheduler
credentials, or private datasets in a public issue. Report a vulnerability
through the repository's private **Security advisories** page. Include the
affected component, reproduction steps, impact, and a suggested mitigation when
available.

The public GitHub Pages demonstration is static. It has no access to the local
research backend, model providers, filesystem, or cluster scheduler.
