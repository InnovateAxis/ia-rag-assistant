"""Step 2.2 — `parse_structure` against real document shapes, fixed as
string literals here rather than read from `corpus/` so a failure points at
the parser, not at a corpus edit changing what these tests see.

The three shapes below are `scripts/generate_corpus.py`'s three distinct
templates: a rate sheet with an inline-value heading, a SOP with a numbered
list, and an MSA amendment (carry-forward F24) where the title itself is the
only thing distinguishing an "Amendment" from a "Master Service Agreement"
sharing an identical body.
"""

from __future__ import annotations

from src.chunking.rules import TableBlock, parse_structure

FUEL_SURCHARGE = """FUEL SURCHARGE POLICY

Effective: 2024-01-01
Tenant: Acme Freight

BASE FUEL SURCHARGE: 15.5%

This surcharge applies to all line-haul charges and is adjusted monthly based on
the national average diesel index.

CALCULATION
Base freight charge × 15.5% = Fuel surcharge

ADJUSTMENT
The surcharge percentage changes on the first of each month based on prevailing
market conditions as reported by the DOT Energy Information Administration.

CUSTOMER NOTIFICATION
Customers receive 24 hours notice of surcharge changes.
"""

DETENTION_SOP = """STANDARD OPERATING PROCEDURE: Detention Procedures

SOP ID: SOP-ACME-001
Revision: 2
Last Updated: 2024-08-15
Tenant: Acme Freight

DETENTION TIME WINDOWS

Free Time: 24 hours
This free time begins when the driver arrives at the facility.

Detention Rates (after free time):
- Hours 0-24: No charge
- Hours 24-48: $50 per 8-hour period
- Hours 48+: $100 per 8-hour period

PROCEDURES
1. Driver initiates detention clock upon arrival
2. Warehouse confirms in system
3. Detention charges accrue automatically after free window
4. Weekly billing to customer account

COMMUNICATION
Customer must be notified if detention will exceed 24 hours.
"""

RATE_SHEET_WITH_ZONES = """Accessorial Charges

Rate Sheet: RS-ACME-002
Published: 2024-09-01
Tenant: Acme Freight

Accessorial Charges RATES

Zone A (0-100 miles):
  LTL: $0.85/lb minimum $85
  Full: $1200 per load

Zone B (100-300 miles):
  LTL: $0.95/lb minimum $120
  Full: $1600 per load

Zone C (300+ miles):
  LTL: $1.10/lb minimum $180
  Full: $2200 per load

Accessorial charges apply for hazmat, oversize, and time-definite services.
"""

MSA_AMENDMENT_TITLED_AMENDMENT = """Amendment

Document ID: MSA-ACME-001
Effective Date: 2024-01-01

OVERVIEW
This document establishes service terms and conditions for freight operations.

SERVICE LEVELS
- Pickup: Within 4 business hours of booking
- Delivery: Per agreed timeline
- Customer Support: 24/7 hotline

LIABILITY
Standard freight liability per DOT regulations.

DISPUTE RESOLUTION
Disputes resolved under Acme Freight's operational guidelines.
"""

MSA_AMENDMENT_TITLED_MSA = MSA_AMENDMENT_TITLED_AMENDMENT.replace(
    "Amendment\n\nDocument ID: MSA-ACME-001",
    "Master Service Agreement\n\nDocument ID: MSA-ACME-002",
)


def test_title_is_the_first_line():
    assert parse_structure(FUEL_SURCHARGE).title == "FUEL SURCHARGE POLICY"
    assert parse_structure(DETENTION_SOP).title == "STANDARD OPERATING PROCEDURE: Detention Procedures"
    assert parse_structure(RATE_SHEET_WITH_ZONES).title == "Accessorial Charges"


def test_metadata_block_is_extracted_as_key_value_pairs():
    parsed = parse_structure(FUEL_SURCHARGE)
    assert parsed.metadata == {"Effective": "2024-01-01", "Tenant": "Acme Freight"}


def test_metadata_extraction_handles_mixed_case_and_abbreviated_labels():
    parsed = parse_structure(DETENTION_SOP)
    assert parsed.metadata == {
        "SOP ID": "SOP-ACME-001",
        "Revision": "2",
        "Last Updated": "2024-08-15",
        "Tenant": "Acme Freight",
    }


def test_all_caps_headings_are_detected_as_section_boundaries():
    parsed = parse_structure(FUEL_SURCHARGE)
    headings = [s.heading for s in parsed.sections]
    assert headings == [
        "BASE FUEL SURCHARGE: 15.5%",
        "CALCULATION",
        "ADJUSTMENT",
        "CUSTOMER NOTIFICATION",
    ]


def test_a_mixed_case_inline_value_line_is_not_mistaken_for_a_heading():
    """"Base freight charge x 15.5% = Fuel surcharge" (mixed case) must stay
    body text under CALCULATION, not be read as its own section."""
    parsed = parse_structure(FUEL_SURCHARGE)
    calculation = next(s for s in parsed.sections if s.heading == "CALCULATION")
    assert any("Fuel surcharge" in b.text for b in calculation.blocks)
    assert [s.heading for s in parsed.sections] != [
        "BASE FUEL SURCHARGE: 15.5%",
        "Base freight charge × 15.5% = Fuel surcharge",
        "CALCULATION",
        "ADJUSTMENT",
        "CUSTOMER NOTIFICATION",
    ]


def test_numbered_list_stays_inside_its_section_not_read_as_a_table():
    """PROCEDURES's "1. Driver initiates..." lines are not indented in the
    source, so they are ordinary paragraph content, not a `TableBlock` —
    only the corpus's indented Zone-style groups are tables."""
    parsed = parse_structure(DETENTION_SOP)
    procedures = next(s for s in parsed.sections if s.heading == "PROCEDURES")
    assert not any(isinstance(b, TableBlock) for b in procedures.blocks)
    assert any("Driver initiates detention clock" in b.text for b in procedures.blocks)


def test_indented_zone_blocks_are_parsed_as_table_blocks_with_their_intro_line():
    parsed = parse_structure(RATE_SHEET_WITH_ZONES)
    tables = [b for s in parsed.sections for b in s.blocks if isinstance(b, TableBlock)]
    assert len(tables) == 3  # Zone A, Zone B, Zone C
    assert tables[0].header_line == "Zone A (0-100 miles):"
    assert tables[0].rows == ("LTL: $0.85/lb minimum $85", "Full: $1200 per load")


def test_msa_amendments_share_an_identical_body_and_differ_only_by_title():
    """Carry-forward F24, confirmed structurally: the parsed body sections
    (headings + text) are byte-identical between the two documents; only
    `.title` differs. Chunking cannot disambiguate them by content — the
    title is the only signal, which is exactly what F24 says."""
    amendment = parse_structure(MSA_AMENDMENT_TITLED_AMENDMENT)
    msa = parse_structure(MSA_AMENDMENT_TITLED_MSA)

    assert amendment.title == "Amendment"
    assert msa.title == "Master Service Agreement"
    assert amendment.title != msa.title
    assert amendment.sections == msa.sections
    assert amendment.metadata != msa.metadata  # Document ID differs (001 vs 002)
