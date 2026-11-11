#!/usr/bin/env bash
# Submit https://neurono-ml.github.io/sciencekit-py/sitemap.xml to Google Search
# Console and Bing Webmaster Tools.
#
# Required environment variables:
#   GOOGLE_ACCESS_TOKEN — OAuth 2.0 bearer token with scope
#                         https://www.googleapis.com/auth/webmasters
#   BING_API_KEY        — Bing Webmaster Tools API key

set -euo pipefail

SITE_URL="https://neurono-ml.github.io/sciencekit-py/"
SITEMAP_URL="https://neurono-ml.github.io/sciencekit-py/sitemap.xml"

missing=()
if [[ -z "${GOOGLE_ACCESS_TOKEN:-}" ]]; then
  missing+=("GOOGLE_ACCESS_TOKEN (OAuth 2.0 token with scope https://www.googleapis.com/auth/webmasters)")
fi
if [[ -z "${BING_API_KEY:-}" ]]; then
  missing+=("BING_API_KEY (from Bing Webmaster Tools -> Settings -> API Access)")
fi

if (( ${#missing[@]} > 0 )); then
  echo "error: cannot submit sitemaps; missing required credential(s):" >&2
  for item in "${missing[@]}"; do
    echo "  - ${item}" >&2
  done
  echo "See docs/seo/search-engines.md for setup instructions." >&2
  exit 1
fi

encoded_site="https%3A%2F%2Fneurono-ml.github.io%2Fsciencekit-py%2F"
encoded_sitemap="https%3A%2F%2Fneurono-ml.github.io%2Fsciencekit-py%2Fsitemap.xml"

echo "Submitting ${SITEMAP_URL} to Google Search Console..."
curl -fsS -X PUT \
  -H "Authorization: Bearer ${GOOGLE_ACCESS_TOKEN}" \
  -H "Content-Length: 0" \
  "https://www.googleapis.com/webmasters/v3/sites/${encoded_site}/sitemaps/${encoded_sitemap}"
echo "Google Search Console submission succeeded."

echo "Submitting ${SITEMAP_URL} to Bing Webmaster Tools..."
curl -fsS -X POST \
  -H "Content-Type: application/json; charset=utf-8" \
  -d "{\"siteUrl\":\"${SITE_URL}\",\"feedUrl\":\"${SITEMAP_URL}\"}" \
  "https://ssl.bing.com/webmaster/api.svc/json/SubmitFeed?apikey=${BING_API_KEY}"
echo
echo "Bing Webmaster Tools submission succeeded."
