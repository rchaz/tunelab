# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in tunelab, please report it privately
via [GitHub Security Advisories](https://github.com/rchaz/tunelab/security/advisories/new).

Do **not** open a public issue for security vulnerabilities.

## Scope

tunelab runs Python scripts locally on your machine via uv. Security concerns include:

- Scripts that could execute arbitrary code beyond their intended purpose
- Dependency declarations that pull unexpected packages
- Data handling that could leak sensitive information

## Response

We will acknowledge reports within 48 hours and aim to release a fix within 7 days for critical issues.
