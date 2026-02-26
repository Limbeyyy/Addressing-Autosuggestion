<?php
/**
 * autocomplete_config.php
 * 
 * PHP proxy that forwards region + language from the frontend form
 * to the Python FastAPI backend's /autocomplete/config endpoint.
 * 
 * Place this file in your web root (e.g. /var/www/html/autocomplete_config.php)
 */

header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: POST, GET, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type');

if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(200);
    exit;
}

// ── Config ────────────────────────────────────────────────────────────────────
define('API_BASE_URL', getenv('AUTOCOMPLETE_API_URL') ?: 'http://127.0.0.1:8000');

// ── Route dispatcher ──────────────────────────────────────────────────────────
$action = $_GET['action'] ?? '';

switch ($action) {
    case 'set_config':
        handle_set_config();
        break;
    case 'get_status':
        handle_get_status();
        break;
    default:
        http_response_code(400);
        echo json_encode(['error' => 'Unknown action. Use ?action=set_config or ?action=get_status']);
}


// ── Handlers ──────────────────────────────────────────────────────────────────

function handle_set_config(): void {
    if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
        http_response_code(405);
        echo json_encode(['error' => 'POST required']);
        return;
    }

    // Accept both JSON body and form POST
    $body = file_get_contents('php://input');
    $data = json_decode($body, true);

    if (!$data) {
        // Fallback: form-encoded POST
        $data = $_POST;
    }

    $region   = trim(strtolower($data['region']   ?? ''));
    $language = trim(strtolower($data['language'] ?? ''));

    if (!$region || !$language) {
        http_response_code(400);
        echo json_encode(['error' => 'Both region and language are required.']);
        return;
    }

    $result = call_api('POST', '/autocomplete/config', [
        'region'   => $region,
        'language' => $language,
    ]);

    http_response_code($result['http_code']);
    echo json_encode($result['body']);
}


function handle_get_status(): void {
    $result = call_api('GET', '/autocomplete/status');
    http_response_code($result['http_code']);
    echo json_encode($result['body']);
}


// ── cURL helper ───────────────────────────────────────────────────────────────

function call_api(string $method, string $path, array $payload = []): array {
    $url = API_BASE_URL . $path;
    $ch  = curl_init($url);

    curl_setopt_array($ch, [
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_TIMEOUT        => 120,          // model init can take a while
        CURLOPT_CONNECTTIMEOUT => 10,
        CURLOPT_HTTPHEADER     => ['Content-Type: application/json', 'Accept: application/json'],
    ]);

    if ($method === 'POST') {
        curl_setopt($ch, CURLOPT_POST, true);
        curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
    }

    $response  = curl_exec($ch);
    $http_code = (int) curl_getinfo($ch, CURLINFO_HTTP_CODE);
    $curl_err  = curl_error($ch);
    curl_close($ch);

    if ($curl_err) {
        return [
            'http_code' => 503,
            'body'      => ['error' => "Could not reach API server: $curl_err"],
        ];
    }

    $decoded = json_decode($response, true);
    if (json_last_error() !== JSON_ERROR_NONE) {
        return [
            'http_code' => 502,
            'body'      => ['error' => 'Invalid JSON response from API', 'raw' => $response],
        ];
    }

    return ['http_code' => $http_code, 'body' => $decoded];
}