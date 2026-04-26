import React from 'react';
import {
  ScrollView,
  TouchableOpacity,
  Text,
  StyleSheet,
  View,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { QueryMode } from '@/services/api';
import { Colors, MODES, Radius, Spacing, Fonts } from '@/constants/theme';

interface Props {
  selected: QueryMode;
  onSelect: (mode: QueryMode) => void;
}

export default function ModeSelector({ selected, onSelect }: Props) {
  return (
    <ScrollView
      horizontal
      showsHorizontalScrollIndicator={false}
      contentContainerStyle={styles.container}
    >
      {MODES.map((m) => {
        const isActive = selected === m.key;
        const color = Colors.modes[m.key];
        return (
          <TouchableOpacity
            key={m.key}
            onPress={() => onSelect(m.key)}
            style={[
              styles.chip,
              { borderColor: color },
              isActive && { backgroundColor: color },
            ]}
            activeOpacity={0.7}
          >
            <Ionicons
              name={m.icon as any}
              size={13}
              color={isActive ? Colors.white : color}
              style={{ marginRight: 4 }}
            />
            <Text style={[styles.label, { color: isActive ? Colors.white : color }]}>
              {m.label}
            </Text>
          </TouchableOpacity>
        );
      })}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.sm,
    gap: Spacing.sm,
    flexDirection: 'row',
  },
  chip: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: Radius.pill,
    borderWidth: 1.5,
  },
  label: {
    fontSize: Fonts.sizes.sm,
    fontWeight: '600',
  },
});
