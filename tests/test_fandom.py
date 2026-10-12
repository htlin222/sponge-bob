from psr.fandom import episode_names, parse_episode_list, parse_transcript

LISTING = ('<tr class="general-header">\n<td rowspan="2"><figure></figure>\n</td>\n<td style="text-align:center"><b>300a</b>\n</td>\n<td style="text-align:left">"<a href="/wiki/X" title="Fake Day">Fake Day</a>" <sup></sup>\n</td>\n<td>2030</td></tr>\n' +
           '<tr class="general-header">\n<td rowspan="2"><figure></figure>\n</td>\n<td style="text-align:center"><b>300b</b>\n</td>\n<td style="text-align:left">"<a href="/wiki/X" title="Other &amp; Day">Other &amp; Day</a>" <sup></sup>\n</td>\n<td>2030</td></tr>\n' +
           '<tr class="general-header">\n<td rowspan="2"><figure></figure>\n</td>\n<td style="text-align:center"><b>301-302</b>\n</td>\n<td style="text-align:left">"<a href="/wiki/X" title="The Long One">The Long One</a>" <sup></sup>\n</td>\n<td>2030</td></tr>\n')


def test_parse_episode_list_reads_prod_and_name():
    assert parse_episode_list(LISTING) == [
        ("300a", "Fake Day"), ("300b", "Other & Day"), ("301-302", "The Long One")]


def test_episode_names_by_production_number_keeps_segment_order():
    listing = parse_episode_list(LISTING)
    assert episode_names(300, listing) == ["Fake Day", "Other & Day"]
    assert episode_names(302, listing) == ["The Long One"]
    assert episode_names(30, listing) == []


def test_parse_transcript_keeps_speakers_and_actions():
    text = """Intro text
{{L|''[A fish walks in.]''}}
{{L|Fish A|Hello, [[Fish B|friend]]! ''[waves]''}}
{{L|Fish B|Bye.}}
[[Category:Transcripts]]"""
    assert parse_transcript(text) == "[A fish walks in.]\nFish A: Hello, friend! [waves]\nFish B: Bye."
