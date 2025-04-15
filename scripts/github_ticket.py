import requests

def create_github_issue(name, email, title, description, repo, token):
    url = f"https://api.github.com/repos/{repo}/issues"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json"
    }

    issue_data = {
        "title": title,
        "body": f"**Submitted by:** {name} ({email})\n\n{description}"
    }

    response = requests.post(url, json=issue_data, headers=headers)

    if response.status_code == 201:
        return True, response.json().get("html_url")
    else:
        return False, response.text
