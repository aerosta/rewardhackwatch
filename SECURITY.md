# Security Policy

## Threat Model & Scope

RewardHackWatch is designed to detect reward hacking and emergent misalignment in **code/task environments** where LLM agents autonomously execute actions.

### In Scope

| Threat | Description |
|--------|-------------|
| Test Manipulation | `sys.exit(0)`, empty tests, mocked results |
| Evaluation Gaming | Exploiting metrics, gaming benchmarks |
| CoT Deception | Hidden intent in chain-of-thought reasoning |
| Reward Tampering | Manipulating reward signals or logging |
| Obfuscation | Learning to hide hacking intent over time |
| Generalization | Hack behavior generalizing to misalignment |

### Out of Scope

| Not Covered | Why |
|-------------|-----|
| Chat Jailbreaks | Use Constitutional AI / RLHF |
| Prompt Injection | Use input validation / sandboxing |
| Social Engineering | Out of scope for code analysis |
| Direct Harm Generation | Use content filters |
| Model Weight Attacks | Requires runtime protections |

### Key Insight

We detect the **transition point** where task-specific reward hacking generalizes into broader misalignment behaviors (alignment faking, oversight subversion, etc.). This is the critical window for intervention.

## Known Limitations

1. **Pattern-Based Detection**: Novel hacking techniques may evade pattern detectors
2. **False Positives**: Some legitimate code patterns may trigger alerts
3. **LLM Judge Reliability**: Judges can be fooled by sophisticated obfuscation
4. **Temporal Analysis**: Requires multiple checkpoints for obfuscation detection
5. **Language Support**: Currently focused on Python code analysis

## Reporting a Vulnerability

If you discover a security vulnerability in RewardHackWatch:

### Do NOT

- Open a public GitHub issue
- Discuss on public forums
- Exploit the vulnerability

### Do

1. **Email**: Send details to [PLACEHOLDER_EMAIL]
2. **Include**:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

3. **Response Time**: We aim to respond within 72 hours

### Disclosure Timeline

1. **Day 0**: Vulnerability reported
2. **Day 1-3**: Initial response and acknowledgment
3. **Day 7-14**: Assessment and fix development
4. **Day 30**: Coordinated disclosure (if applicable)

## Security Best Practices

When using RewardHackWatch:

### API Keys

```bash
# Use environment variables
export ANTHROPIC_API_KEY="your-key"

# Never commit to git
echo ".env" >> .gitignore
```

### Production Deployment

1. **Isolate**: Run in sandboxed environment
2. **Limit**: Restrict network access
3. **Monitor**: Log all alert triggers
4. **Review**: Human review of CRITICAL alerts

### Webhook Security

If using webhook alerts:

```python
config = MonitorConfig(
    enable_webhook=True,
    webhook_url="https://your-secure-endpoint.com/alerts",
)
```

- Use HTTPS only
- Implement webhook authentication
- Rate limit incoming requests

## Dependencies

We minimize dependencies and keep them updated:

```bash
# Check for vulnerabilities
pip-audit

# Update dependencies
pip install --upgrade -e ".[dev]"
```

## Acknowledgments

We thank the following for security research contributions:

- [Your acknowledgments here]

## Contact

- Security issues: [PLACEHOLDER_EMAIL]
- General questions: Open a GitHub issue
