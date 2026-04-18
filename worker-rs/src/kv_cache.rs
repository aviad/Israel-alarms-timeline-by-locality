/// kv_cache.rs — KV store helpers with optional zlib compression.
///
/// All helpers are thin wrappers around the workers-rs KV API.
/// The zlib variants store data as base64-encoded text (not raw bytes)
/// because wrangler's local KV simulation stores Vec<u8> as a JSON array
/// of integers rather than as raw binary, which corrupts binary KV round-trips.
/// Base64 text is safe in both local dev and production.
/// Key prefix "rs:" avoids mixed reads with the Python worker.

use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use std::io::{Read, Write};
use worker::kv::KvStore;
use worker::{Error, Result};

fn kv_err(e: impl std::fmt::Display) -> Error {
    Error::RustError(e.to_string())
}

// ── Plain text ───────────────────────────────────────────────────────────────

pub async fn get_text(kv: &KvStore, key: &str) -> Result<Option<String>> {
    kv.get(key).text().await.map_err(kv_err)
}

pub async fn put_text(kv: &KvStore, key: &str, value: &str, ttl_secs: u64) -> Result<()> {
    kv.put(key, value).map_err(kv_err)?.expiration_ttl(ttl_secs).execute().await.map_err(kv_err)
}

// ── Raw bytes ────────────────────────────────────────────────────────────────

pub async fn get_bytes(kv: &KvStore, key: &str) -> Result<Option<Vec<u8>>> {
    kv.get(key).bytes().await.map_err(kv_err)
}

pub async fn put_bytes(kv: &KvStore, key: &str, value: Vec<u8>, ttl_secs: u64) -> Result<()> {
    kv.put(key, value).map_err(kv_err)?.expiration_ttl(ttl_secs).execute().await.map_err(kv_err)
}

// ── Zlib-compressed bytes ────────────────────────────────────────────────────

/// Decompress a zlib blob into raw bytes.
pub fn zlib_decompress(data: &[u8]) -> Result<Vec<u8>> {
    let mut dec = ZlibDecoder::new(data);
    let mut out = Vec::new();
    dec.read_to_end(&mut out)
        .map_err(|e| Error::RustError(format!("zlib decompress: {e}")))?;
    Ok(out)
}

/// Compress raw bytes with zlib (default level).
pub fn zlib_compress(data: &[u8]) -> Result<Vec<u8>> {
    let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
    enc.write_all(data)
        .map_err(|e| Error::RustError(format!("zlib compress write: {e}")))?;
    enc.finish()
        .map_err(|e| Error::RustError(format!("zlib compress finish: {e}")))
}

/// Fetch a zlib-compressed blob from KV (stored as base64 text) and decompress it.
pub async fn get_zlib(kv: &KvStore, key: &str) -> Result<Option<Vec<u8>>> {
    match get_text(kv, key).await? {
        None => Ok(None),
        Some(b64) => {
            let compressed = b64_decode(b64.trim())
                .map_err(|e| Error::RustError(format!("zlib kv base64: {e}")))?;
            Ok(Some(zlib_decompress(&compressed)?))
        }
    }
}

/// Compress bytes and store them as base64 text in KV with the given TTL.
pub async fn put_zlib(kv: &KvStore, key: &str, data: &[u8], ttl_secs: u64) -> Result<()> {
    let compressed = zlib_compress(data)?;
    let b64 = b64_encode(&compressed);
    put_text(kv, key, &b64, ttl_secs).await
}

// ── Minimal base64 (RFC 4648 standard alphabet, no padding stripped) ──────────

const B64_CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn b64_encode(input: &[u8]) -> String {
    let mut out = Vec::with_capacity((input.len() + 2) / 3 * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as usize;
        let b1 = if chunk.len() > 1 { chunk[1] as usize } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as usize } else { 0 };
        let combined = (b0 << 16) | (b1 << 8) | b2;
        out.push(B64_CHARS[(combined >> 18) & 0x3f]);
        out.push(B64_CHARS[(combined >> 12) & 0x3f]);
        out.push(if chunk.len() > 1 { B64_CHARS[(combined >> 6) & 0x3f] } else { b'=' });
        out.push(if chunk.len() > 2 { B64_CHARS[combined & 0x3f]        } else { b'=' });
    }
    // Safety: output contains only ASCII b64 chars and '='
    unsafe { String::from_utf8_unchecked(out) }
}

fn b64_decode(input: &str) -> std::result::Result<Vec<u8>, &'static str> {
    let bytes = input.as_bytes();
    if bytes.len() % 4 != 0 {
        return Err("base64 length not multiple of 4");
    }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for chunk in bytes.chunks(4) {
        let c: Vec<u8> = chunk.iter().map(|&b| b64_val(b)).collect();
        if c[0] == 255 || c[1] == 255 { return Err("invalid base64 char"); }
        out.push((c[0] << 2) | (c[1] >> 4));
        if c[2] != 64 { // not '='
            if c[2] == 255 { return Err("invalid base64 char"); }
            out.push((c[1] << 4) | (c[2] >> 2));
        }
        if c[3] != 64 {
            if c[3] == 255 { return Err("invalid base64 char"); }
            out.push((c[2] << 6) | c[3]);
        }
    }
    Ok(out)
}

/// Map a base64 character to its 6-bit value; 64 = padding '='; 255 = invalid.
fn b64_val(b: u8) -> u8 {
    match b {
        b'A'..=b'Z' => b - b'A',
        b'a'..=b'z' => b - b'a' + 26,
        b'0'..=b'9' => b - b'0' + 52,
        b'+'        => 62,
        b'/'        => 63,
        b'='        => 64,
        _           => 255,
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_compress_decompress() {
        let original = b"hello world! this is some test data for compression 1234567890";
        let compressed = zlib_compress(original).unwrap();
        assert!(compressed.len() < original.len() + 20);
        let decompressed = zlib_decompress(&compressed).unwrap();
        assert_eq!(decompressed, original);
    }

    #[test]
    fn roundtrip_empty() {
        let compressed = zlib_compress(b"").unwrap();
        let decompressed = zlib_decompress(&compressed).unwrap();
        assert_eq!(decompressed, b"");
    }

    #[test]
    fn reject_corrupt_payload() {
        let corrupt = b"this is not a valid zlib stream at all";
        assert!(zlib_decompress(corrupt).is_err());
    }

    /// Verify Rust can decompress a blob that was compressed by Python's zlib.
    /// The fixture bytes were produced with: import zlib; zlib.compress(b"hello from python").
    #[test]
    fn compat_python_zlib() {
        let python_blob: &[u8] = &[
            0x78, 0x9c, 0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x57, 0x48, 0x2b, 0xca, 0xcf, 0x55, 0x28,
            0xa8, 0x2c, 0xc9, 0xc8, 0xcf, 0x03, 0x00, 0x3a, 0xfe, 0x06, 0xab,
        ];
        let decompressed = zlib_decompress(python_blob).unwrap();
        assert_eq!(decompressed, b"hello from python");
    }

    // ── base64 tests ──────────────────────────────────────────────────────────

    #[test]
    fn b64_encode_decode_roundtrip() {
        let cases: &[&[u8]] = &[b"", b"f", b"fo", b"foo", b"foob", b"fooba", b"foobar"];
        for &input in cases {
            let encoded = b64_encode(input);
            let decoded = b64_decode(&encoded).unwrap();
            assert_eq!(decoded, input, "roundtrip failed for {:?}", input);
        }
    }

    #[test]
    fn b64_known_vectors() {
        // RFC 4648 test vectors
        assert_eq!(b64_encode(b""),       "");
        assert_eq!(b64_encode(b"f"),      "Zg==");
        assert_eq!(b64_encode(b"fo"),     "Zm8=");
        assert_eq!(b64_encode(b"foo"),    "Zm9v");
        assert_eq!(b64_encode(b"foob"),   "Zm9vYg==");
        assert_eq!(b64_encode(b"fooba"),  "Zm9vYmE=");
        assert_eq!(b64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn b64_decode_invalid() {
        assert!(b64_decode("!!!").is_err()); // wrong length
        assert!(b64_decode("!!!A").is_err()); // invalid char
    }

    #[test]
    fn zlib_b64_roundtrip() {
        // Simulates what put_zlib/get_zlib do without KV I/O
        let original = b"hello world this is test data 0123456789";
        let compressed = zlib_compress(original).unwrap();
        let b64 = b64_encode(&compressed);
        let decoded = b64_decode(&b64).unwrap();
        let decompressed = zlib_decompress(&decoded).unwrap();
        assert_eq!(decompressed, original);
    }
}
