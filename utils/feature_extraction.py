# ============================================================
# PhishGuard — Feature Extraction Module
# utils/feature_extraction.py
#
# Converts a raw URL string into a numerical feature vector
# that the machine learning model can understand and classify.
# ============================================================

import re
import math
import urllib.parse

# ─── Suspicious keywords commonly found in phishing URLs ───────────────────
SUSPICIOUS_KEYWORDS = [
    'login', 'verify', 'secure', 'bank', 'update', 'password',
    'account', 'signin', 'billing', 'confirm', 'paypal', 'ebay',
    'amazon', 'apple', 'microsoft', 'support', 'service', 'urgent',
    'alert', 'suspend', 'locked', 'restore', 'validate', 'credential'
]

# ─── Suspicious TLDs often used in phishing campaigns ──────────────────────
SUSPICIOUS_TLDS = ['.tk', '.ml', '.ga', '.cf', '.xyz', '.top', '.gq', '.pw']


def has_ip_address(url: str) -> int:
    """
    Checks if the URL contains an IP address instead of a proper domain name.
    Phishing sites often use raw IP addresses to hide their identity.

    Returns 1 if IP address found, 0 otherwise.
    """
    ip_pattern = r'(http[s]?://)?(\d{1,3}\.){3}\d{1,3}'
    return 1 if re.search(ip_pattern, url) else 0


def calculate_entropy(text: str) -> float:
    """
    Calculates Shannon entropy of the URL string.
    High entropy = more random characters = more likely to be obfuscated/malicious.

    Formula: H = -Σ p(x) * log2(p(x))
    """
    if not text:
        return 0.0
    freq = {}
    for char in text:
        freq[char] = freq.get(char, 0) + 1
    entropy = 0.0
    length = len(text)
    for count in freq.values():
        probability = count / length
        if probability > 0:
            entropy -= probability * math.log2(probability)
    return round(entropy, 4)


def has_suspicious_tld(url: str) -> int:
    """
    Checks if the URL uses a TLD commonly associated with free/phishing domains.
    Returns 1 if suspicious TLD found, 0 otherwise.
    """
    url_lower = url.lower()
    return 1 if any(tld in url_lower for tld in SUSPICIOUS_TLDS) else 0


def extract_features(url: str) -> dict:
    """
    Main feature extraction function.
    Takes a URL string and returns a dictionary of 22 numerical features.

    Parameters:
        url (str): The full URL to analyze

    Returns:
        dict: Feature name → numeric value mapping
    """
    url = url.strip()

    # ── Parse URL into components ──────────────────────────
    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc        # e.g. "www.example.com"
        path   = parsed.path          # e.g. "/login/verify"
        query  = parsed.query         # e.g. "id=123&token=abc"
    except Exception:
        domain = ''
        path   = ''
        query  = ''

    # ─────────────────────────────────────────────────────
    # FEATURE GROUP 1: Basic URL structure metrics
    # ─────────────────────────────────────────────────────
    url_length   = len(url)
    num_dots     = url.count('.')
    num_hyphens  = url.count('-')
    num_digits   = sum(c.isdigit() for c in url)
    num_slash    = url.count('/')
    num_at       = url.count('@')       # @ can redirect to different host
    num_question = url.count('?')
    num_equal    = url.count('=')
    num_percent  = url.count('%')       # URL encoding - hides malicious content
    num_ampersand= url.count('&')

    # ─────────────────────────────────────────────────────
    # FEATURE GROUP 2: Security & protocol indicators
    # ─────────────────────────────────────────────────────
    has_https    = 1 if url.lower().startswith('https') else 0
    has_ip       = has_ip_address(url)

    # Check for non-standard port (e.g. :8080, :4443) — excludes standard 80/443
    has_port = 0
    if domain:
        port_match = re.search(r':(\d+)$', domain)
        if port_match:
            port_num = int(port_match.group(1))
            if port_num not in (80, 443):
                has_port = 1

    # Double slash after protocol = possible redirect trick (e.g. http://evil.com//real.com)
    url_body = url[8:] if url.startswith('https://') else url[7:] if url.startswith('http://') else url
    has_double_slash = 1 if '//' in url_body else 0

    # ─────────────────────────────────────────────────────
    # FEATURE GROUP 3: Domain & subdomain analysis
    # ─────────────────────────────────────────────────────
    # Count subdomains (legitimate sites rarely have 3+ subdomains)
    domain_clean   = domain.replace('www.', '').split(':')[0]   # strip port too
    subdomain_count = max(0, len(domain_clean.split('.')) - 2)  # -2 for name.tld

    # Suspicious TLD detection
    suspicious_tld = has_suspicious_tld(url)

    # ─────────────────────────────────────────────────────
    # FEATURE GROUP 4: Keyword-based analysis
    # ─────────────────────────────────────────────────────
    url_lower = url.lower()
    keyword_count            = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url_lower)
    has_suspicious_keyword   = 1 if keyword_count > 0 else 0

    # ─────────────────────────────────────────────────────
    # FEATURE GROUP 5: Complexity & obfuscation metrics
    # ─────────────────────────────────────────────────────
    url_entropy  = calculate_entropy(url)
    path_length  = len(path)
    query_length = len(query)

    # URL depth = number of path segments (e.g. /a/b/c → depth 3)
    url_depth = len([p for p in path.split('/') if p])

    # ─────────────────────────────────────────────────────
    # Return all features as a dictionary
    # Order matters! Must match the column order used during training.
    # ─────────────────────────────────────────────────────
    features = {
        'url_length'            : url_length,
        'num_dots'              : num_dots,
        'num_hyphens'           : num_hyphens,
        'num_digits'            : num_digits,
        'has_https'             : has_https,
        'has_ip'                : has_ip,
        'num_at'                : num_at,
        'num_question'          : num_question,
        'num_equal'             : num_equal,
        'num_slash'             : num_slash,
        'num_percent'           : num_percent,
        'num_ampersand'         : num_ampersand,
        'subdomain_count'       : subdomain_count,
        'has_suspicious_keyword': has_suspicious_keyword,
        'keyword_count'         : keyword_count,
        'url_entropy'           : url_entropy,
        'path_length'           : path_length,
        'query_length'          : query_length,
        'has_port'              : has_port,
        'has_double_slash'      : has_double_slash,
        'url_depth'             : url_depth,
        'suspicious_tld'        : suspicious_tld,
    }

    return features


def get_feature_vector(url: str) -> list:
    """
    Returns features as a plain Python list (for model.predict).
    The order is fixed — must match FEATURE_NAMES below.
    """
    return list(extract_features(url).values())


def explain_features(features: dict, url: str) -> list[str]:
    """
    Generates human-readable warning messages explaining WHY a URL
    is considered suspicious. Used in the dashboard explanation panel.

    Parameters:
        features (dict): Output of extract_features()
        url (str): Original URL string

    Returns:
        list of warning strings
    """
    warnings = []

    if features['has_https'] == 0:
        warnings.append("🔓 No HTTPS — the connection is NOT encrypted (plain HTTP)")

    if features['has_ip'] == 1:
        warnings.append("🖥️  IP address used instead of a proper domain name")

    if features['url_length'] > 75:
        warnings.append(f"📏 Unusually long URL ({features['url_length']} chars) — typical of phishing")

    if features['num_hyphens'] > 3:
        warnings.append(f"➖ High number of hyphens ({features['num_hyphens']}) — domain spoofing pattern")

    if features['has_suspicious_keyword'] == 1:
        found = [kw for kw in SUSPICIOUS_KEYWORDS if kw in url.lower()]
        warnings.append(f"🔑 Suspicious keyword(s) detected: {', '.join(found[:4])}")

    if features['num_dots'] > 4:
        warnings.append(f"🔵 Excessive dots ({features['num_dots']}) — possible subdomain abuse")

    if features['num_at'] > 0:
        warnings.append("⚠️  '@' symbol found — used to trick browsers into loading a different host")

    if features['has_double_slash'] == 1:
        warnings.append("↩️  Double slash (//) after protocol — possible redirect attack")

    if features['num_digits'] > 12:
        warnings.append(f"🔢 High digit count ({features['num_digits']}) — possible encoded/random domain")

    if features['keyword_count'] > 2:
        warnings.append(f"🚨 Multiple suspicious keywords found ({features['keyword_count']}) — strong phishing signal")

    if features['has_port'] == 1:
        warnings.append("🔌 Non-standard port in URL — legitimate sites use 80/443")

    if features['suspicious_tld'] == 1:
        warnings.append("🌐 Uses a free/suspicious TLD (.tk .ml .xyz etc.) — common in phishing campaigns")

    if features['subdomain_count'] > 2:
        warnings.append(f"🌿 {features['subdomain_count']} nested subdomains — legitimate sites rarely need this many")

    if features['num_percent'] > 3:
        warnings.append(f"🔏 Heavy URL encoding ({features['num_percent']} % chars) — possible content obfuscation")

    if features['url_entropy'] > 4.5:
        warnings.append(f"🎲 High URL entropy ({features['url_entropy']:.2f}) — randomised/obfuscated characters detected")

    # If nothing suspicious is found, add a positive note
    if not warnings:
        warnings.append("✅ No obvious suspicious patterns detected in URL structure")

    return warnings


# ── Feature name list (for model training and display) ──────────────────────
FEATURE_NAMES = list(extract_features("https://example.com").keys())
