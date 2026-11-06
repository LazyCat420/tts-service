"""Port of SAM's renderer: frame building and sample rendering.
Stdlib only; tables live in tables.py."""
import math
import re

from tables import TABLES as _T

sampledConsonantValues0 = _T['sampledConsonantValues0']
stressPitch_tab47492 = _T['stressPitch_tab47492']
blendRank = _T['blendRank']
outBlendLength = _T['outBlendLength']
inBlendLength = _T['inBlendLength']
sampledConsonantFlags = _T['sampledConsonantFlags']
frequencyData = _T['frequencyData']
ampldata = _T['ampldata']
sampleTable = _T['sampleTable']

PHONEME_PERIOD = 1
PHONEME_QUESTION = 2


def set_mouth_throat(mouth, throat):
    def trans(factor, f):
        return (((factor * f) >> 8) & 0xFF) << 1
    fd = [[v & 0xFF for v in frequencyData],
          [(v >> 8) & 0xFF for v in frequencyData],
          [(v >> 16) & 0xFF for v in frequencyData]]
    for pos in list(range(5, 30)) + list(range(48, 54)):
        fd[0][pos] = trans(mouth, fd[0][pos])
        fd[1][pos] = trans(throat, fd[1][pos])
    return fd


def _add_inflection(inflection, pos, pitches):
    end = pos
    pos = 0 if pos < 30 else pos - 30
    A = pitches[pos]
    while A == 127:
        pos += 1
        A = pitches[pos]
    while pos != end:
        A += inflection
        pitches[pos] = A & 0xFF
        pos += 1
        while pos != end and pitches[pos] == 255:
            pos += 1


def create_frames(pitch, tuples, fd):
    pitches = []
    frequency = [[], [], []]
    amplitude = [[], [], []]
    flags = []
    X = 0
    for tp in tuples:
        ph = tp[0]
        if ph == PHONEME_PERIOD:
            _add_inflection(1, X, pitches)
        elif ph == PHONEME_QUESTION:
            _add_inflection(255, X, pitches)
        phase1 = stressPitch_tab47492[tp[2]]
        for _ in range(tp[1]):
            frequency[0].append(fd[0][ph])
            frequency[1].append(fd[1][ph])
            frequency[2].append(fd[2][ph])
            a = ampldata[ph]
            amplitude[0].append(a & 0xFF)
            amplitude[1].append((a >> 8) & 0xFF)
            amplitude[2].append((a >> 16) & 0xFF)
            flags.append(sampledConsonantFlags[ph])
            pitches.append((pitch + phase1) & 0xFF)
            X += 1
    return pitches, frequency, amplitude, flags


def _jsdiv_trunc(a, b):
    return int(a / b)


def create_transitions(pitches, frequency, amplitude, tuples):
    tables = [pitches, frequency[0], frequency[1], frequency[2],
              amplitude[0], amplitude[1], amplitude[2]]

    def interpolate(width, table, frame, change):
        sign = change < 0
        remainder = abs(change) % width
        div = _jsdiv_trunc(change, width)
        error = 0
        pos = width
        tb = tables[table]
        while True:
            pos -= 1
            if not pos > 0:
                break
            val = tb[frame] + div
            error += remainder
            if error >= width:
                error -= width
                if sign:
                    val -= 1
                elif val:
                    val += 1
            frame += 1
            while len(tb) <= frame:
                tb.append(0)
            tb[frame] = val
            val += div

    boundary = 0
    for pos in range(len(tuples) - 1):
        phoneme = tuples[pos][0]
        nxt = tuples[pos + 1][0]
        next_rank = blendRank[nxt]
        rank = blendRank[phoneme]
        if rank == next_rank:
            out_bf = outBlendLength[phoneme]
            in_bf = outBlendLength[nxt]
        elif rank < next_rank:
            out_bf = inBlendLength[nxt]
            in_bf = outBlendLength[nxt]
        else:
            out_bf = outBlendLength[phoneme]
            in_bf = inBlendLength[phoneme]
        boundary += tuples[pos][1]
        trans_end = boundary + in_bf
        trans_start = boundary - out_bf
        trans_length = out_bf + in_bf
        if ((trans_length - 2) & 128) == 0:
            cur_width = tuples[pos][1] >> 1
            next_width = tuples[pos + 1][1] >> 1
            p = pitches[boundary + next_width] - pitches[boundary - cur_width]
            interpolate(cur_width + next_width, 0, trans_start, p)
            for table in range(1, 7):
                value = tables[table][trans_end] - tables[table][trans_start]
                interpolate(trans_length, table, trans_start, value)
    return boundary + tuples[-1][1]


_AMPL_RESCALE = [0x00, 0x01, 0x02, 0x02, 0x02, 0x03, 0x03, 0x04,
                 0x04, 0x05, 0x06, 0x08, 0x09, 0x0B, 0x0D, 0x0F]


def prepare_frames(data, pitch, mouth, throat, singmode=False):
    fd = set_mouth_throat(mouth, throat)
    pitches, frequency, amplitude, flags = create_frames(pitch, data, fd)
    t = create_transitions(pitches, frequency, amplitude, data)
    if not singmode:
        for i in range(len(pitches)):
            # JS: undefined >> 1 == 0 for transition overrun past frame count
            pitches[i] -= (frequency[0][i] >> 1) if i < len(frequency[0]) else 0
    for i in range(len(amplitude[0]) - 1, -1, -1):
        for k in range(3):
            amplitude[k][i] = _AMPL_RESCALE[amplitude[k][i]]
    return t, frequency, pitches, amplitude, flags


_TIMETABLE = [
    [162, 167, 167, 127, 128],
    [226, 60, 60, 0, 0],
    [225, 60, 59, 0, 0],
    [200, 0, 0, 54, 55],
    [199, 0, 0, 54, 54],
]


class OutputBuffer:
    def __init__(self, size):
        self.size = size
        self.buf = bytearray(size + 8)
        self.pos = 0
        self.old = 0

    def ary(self, index, arr):
        self.pos += _TIMETABLE[self.old][index]
        if self.pos // 50 > self.size:
            raise RuntimeError('Buffer overflow')
        self.old = index
        base = self.pos // 50
        for k in range(5):
            if base + k < self.size:
                self.buf[base + k] = arr[k] & 0xFF

    def __call__(self, index, A):
        s = (A & 15) * 16
        self.ary(index, [s] * 5)

    def get(self):
        return bytes(self.buf[:min(self.pos // 50, self.size)])


def _render_sample(out, last_off, flag, pitch):
    kind = (flag & 7) - 1
    page = (kind * 256) & 0xFFFF
    off = flag & 248

    def rs(i1, v1, i0, v0):
        sample = sampleTable[page + off]
        for _ in range(8):
            if sample & 128:
                out(i1, v1)
            else:
                out(i0, v0)
            sample <<= 1

    if off == 0:
        phase1 = (pitch >> 4) ^ 255
        off = last_off & 0xFF
        while True:
            rs(3, 26, 4, 6)
            off = (off + 1) & 0xFF
            phase1 += 1
            if not (phase1 & 0xFF):
                break
        return off
    off = off ^ 255
    value0 = sampledConsonantValues0[kind] & 0xFF
    while True:
        rs(2, 5, 1, value0)
        off += 1
        if not (off & 0xFF):
            break
    return last_off


def _sinus(x):
    return int(math.sin(2 * math.pi * (x / 256)) * 127)


def _get(arr, i):
    """JS reads past the end of these arrays as `undefined`, which behaves as 0
    in the bit/compare ops used below. Mirror that instead of raising."""
    return arr[i] if 0 <= i < len(arr) else 0


def process_frames(out, frame_count, speed, frequency, pitches, amplitude, flagsarr):
    speedcounter = speed
    phase1 = phase2 = phase3 = 0
    last_off = 0
    pos = 0
    glottal = _get(pitches, 0)
    mem38 = int(glottal * .75)
    while frame_count:
        flags = _get(flagsarr, pos)
        if (flags & 248) != 0:
            last_off = _render_sample(out, last_off, flags, _get(pitches, pos & 0xFF))
            pos += 2
            frame_count -= 2
            speedcounter = speed
        else:
            ary = []
            p1 = phase1 * 256
            p2 = phase2 * 256
            p3 = phase3 * 256
            for k in range(5):
                sp1 = _sinus(0xFF & (p1 >> 8))
                sp2 = _sinus(0xFF & (p2 >> 8))
                rp3 = -0x70 if (0xFF & (p3 >> 8)) < 129 else 0x70
                sin1 = sp1 * (_get(amplitude[0], pos) & 0x0F)
                sin2 = sp2 * (_get(amplitude[1], pos) & 0x0F)
                rect = rp3 * (_get(amplitude[2], pos) & 0x0F)
                mux = (sin1 + sin2 + rect) / 32
                mux += 128
                ary.append(int(mux))
                p1 += _get(frequency[0], pos) * 256 // 4
                p2 += _get(frequency[1], pos) * 256 // 4
                p3 += _get(frequency[2], pos) * 256 // 4
            out.ary(0, ary)
            speedcounter -= 1
            if speedcounter == 0:
                pos += 1
                frame_count -= 1
                if frame_count == 0:
                    return
                speedcounter = speed
            glottal -= 1
            if glottal != 0:
                mem38 -= 1
                if mem38 != 0 or flags == 0:
                    phase1 += _get(frequency[0], pos)
                    phase2 += _get(frequency[1], pos)
                    phase3 += _get(frequency[2], pos)
                    continue
                last_off = _render_sample(out, last_off, flags, _get(pitches, pos & 0xFF))
        glottal = _get(pitches, pos)
        mem38 = int(glottal * .75)
        phase1 = phase2 = phase3 = 0


def render(data, pitch=64, speed=72, mouth=128, throat=128, singmode=False):
    pitch &= 0xFF
    mouth &= 0xFF
    throat &= 0xFF
    speed = (speed or 72) & 0xFF
    t, frequency, pitches, amplitude, flags = prepare_frames(
        data, pitch, mouth, throat, singmode)
    out = OutputBuffer(int(176.4 * sum(v[1] for v in data) * speed))
    process_frames(out, t, speed, frequency, pitches, amplitude, flags)
    return out.get()
