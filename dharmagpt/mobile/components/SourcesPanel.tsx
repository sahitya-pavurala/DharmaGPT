import React, { useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Linking,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { SourceChunk } from '@/services/api';
import { Colors, Fonts, Spacing, Radius } from '@/constants/theme';

interface Props {
  sources: SourceChunk[];
}

export default function SourcesPanel({ sources }: Props) {
  const [expanded, setExpanded] = useState<number | null>(null);

  return (
    <View style={styles.container}>
      {sources.map((src, i) => {
        const isOpen = expanded === i;
        const isAudio = src.source_type === 'audio';

        return (
          <TouchableOpacity
            key={i}
            style={styles.card}
            onPress={() => setExpanded(isOpen ? null : i)}
            activeOpacity={0.8}
          >
            {/* Header row */}
            <View style={styles.headerRow}>
              <View style={styles.badgeRow}>
                <Ionicons
                  name={isAudio ? 'musical-notes' : 'book'}
                  size={11}
                  color={isAudio ? Colors.lotus : Colors.saffron}
                />
                <Text style={[styles.badge, { color: isAudio ? Colors.lotus : Colors.saffron }]}>
                  {isAudio ? 'Audio' : 'Text'}
                </Text>
                {src.section && (
                  <Text style={styles.section} numberOfLines={1}>
                    {src.section}
                  </Text>
                )}
              </View>
              <Ionicons
                name={isOpen ? 'chevron-up' : 'chevron-down'}
                size={13}
                color={Colors.ash}
              />
            </View>

            {/* Citation */}
            <Text style={styles.citation} numberOfLines={isOpen ? undefined : 1}>
              {src.citation}
            </Text>

            {/* Metadata chips */}
            {(src.chapter || src.verse) && (
              <View style={styles.metaRow}>
                {src.chapter && (
                  <View style={styles.metaChip}>
                    <Text style={styles.metaText}>Ch. {src.chapter}</Text>
                  </View>
                )}
                {src.verse && (
                  <View style={styles.metaChip}>
                    <Text style={styles.metaText}>v. {src.verse}</Text>
                  </View>
                )}
                <View style={styles.metaChip}>
                  <Text style={styles.metaText}>{Math.round(src.score * 100)}% match</Text>
                </View>
              </View>
            )}

            {/* Expanded text */}
            {isOpen && (
              <View style={styles.expandedBlock}>
                <Text style={styles.sourceText}>{src.text}</Text>
                {src.audio_timestamp && (
                  <Text style={styles.timestamp}>⏱ {src.audio_timestamp}</Text>
                )}
                {src.url && (
                  <TouchableOpacity onPress={() => Linking.openURL(src.url!)}>
                    <Text style={styles.link}>Open source ↗</Text>
                  </TouchableOpacity>
                )}
              </View>
            )}
          </TouchableOpacity>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    marginTop: Spacing.sm,
    gap: 6,
  },
  card: {
    backgroundColor: Colors.darkBg,
    borderRadius: Radius.md,
    padding: Spacing.sm,
    borderWidth: 1,
    borderColor: Colors.darkBorder,
  },
  headerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  badgeRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    flex: 1,
  },
  badge: {
    fontSize: Fonts.sizes.xs,
    fontWeight: '700',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  section: {
    fontSize: Fonts.sizes.xs,
    color: Colors.ash,
    flex: 1,
  },
  citation: {
    fontSize: Fonts.sizes.sm,
    color: Colors.sandal,
    fontStyle: 'italic',
  },
  metaRow: {
    flexDirection: 'row',
    gap: 6,
    marginTop: 6,
    flexWrap: 'wrap',
  },
  metaChip: {
    backgroundColor: Colors.darkCard,
    paddingHorizontal: 7,
    paddingVertical: 2,
    borderRadius: Radius.pill,
    borderWidth: 1,
    borderColor: Colors.darkBorder,
  },
  metaText: {
    fontSize: Fonts.sizes.xs,
    color: Colors.ash,
  },
  expandedBlock: {
    marginTop: Spacing.sm,
    paddingTop: Spacing.sm,
    borderTopWidth: 1,
    borderTopColor: Colors.darkBorder,
  },
  sourceText: {
    fontSize: Fonts.sizes.sm,
    color: Colors.cream,
    lineHeight: 20,
  },
  timestamp: {
    fontSize: Fonts.sizes.xs,
    color: Colors.ash,
    marginTop: 6,
  },
  link: {
    fontSize: Fonts.sizes.sm,
    color: Colors.saffron,
    marginTop: 6,
    fontWeight: '600',
  },
});
