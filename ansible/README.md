# Ansible Deployment

This directory contains Ansible playbooks for deploying AI Asylum to remote servers.

## Prerequisites

1. Install Ansible:
   ```bash
   pip install ansible
   ```

2. Configure inventory:
   - Edit `inventory.ini` with your server details
   - Set SSH keys and connection details

## Usage

### Basic Deployment

```bash
# Deploy to production
ansible-playbook -i inventory.ini playbook.yml --limit production

# Deploy to staging
ansible-playbook -i inventory.ini playbook.yml --limit staging

# Deploy to localhost (for testing)
ansible-playbook -i inventory.ini playbook.yml --limit local
```

### With Extra Variables

```bash
ansible-playbook -i inventory.ini playbook.yml \
  --extra-vars "openai_api_key=sk-... \
                api_secret_key=your-secret \
                server_name=api.yourdomain.com"
```

### Using Ansible Vault for Secrets

1. Create encrypted variables file:
   ```bash
   ansible-vault create group_vars/production/secrets.yml
   ```

2. Add secrets:
   ```yaml
   openai_api_key: sk-...
   api_secret_key: your-secret
   ```

3. Run playbook:
   ```bash
   ansible-playbook -i inventory.ini playbook.yml \
     --limit production \
     --ask-vault-pass
   ```

## Variables

Key variables (set in inventory or via --extra-vars):

- `git_repo`: Repository URL
- `git_branch`: Branch to deploy (default: main)
- `database_url`: PostgreSQL connection string
- `openai_api_key`: OpenAI API key
- `api_secret_key`: API secret key
- `server_name`: Domain name for Nginx
- `cors_origins`: CORS allowed origins

## Post-Deployment

After deployment:

1. Check service status:
   ```bash
   systemctl status aiasylum
   ```

2. Check logs:
   ```bash
   journalctl -u aiasylum -f
   ```

3. Test API:
   ```bash
   curl http://your-server/api/v1/test-runs/
   ```

## Troubleshooting

- Ensure PostgreSQL is running: `systemctl status postgresql`
- Check Nginx configuration: `nginx -t`
- Review application logs: `journalctl -u aiasylum -n 100`
