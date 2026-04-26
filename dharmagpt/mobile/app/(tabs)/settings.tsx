import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
  Switch,
  Linking,
  ActivityIndicator,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useSettingsStore } from '@/store/settingsStore';
import { Colors, Fonts, Spacing, Radius, MODES, LANGUAGES } from '@/constants/theme';
import { dharmaApi, HealthResponse } from '@/services/api';
import { QueryMode } from '@/services/api';

function SectionHeader({ title }: { title: string }) {
  return <Text style={styles.sectionHeader}>{title}</Text>;
}

function Row({ children }: { children: React.ReactNode }) {
  return <View style={styles.row}>{children}</View>;
}

export default function SettingsScreen() {
  const {
    language,
    defaultMode,
    showSources,
    hapticFeedback,
    setLanguage,
    setDefaultMode,
    setShowSources,
    setHapticFeedback,
  } = useSettingsStore();

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState(false);

  const checkHealth = async () => {
    setHealthLoading(true);
    try {
      const h = await dharmaApi.health();
      setHealth(h);
    } catch {
      setHealth(null);
    } finally {
      setHealthLoading(false);
    }
  };

  useEffect(() => {
    checkHealth();
  }, []);

  return (
    <SafeAreaView style={styles.safe} edges={['bottom']}>
      <ScrollView contentContainerStyle={styles.content}>

        {/* ── Language ─────────────────────────────────── */}
        <SectionHeader title="Language" />
        <View style={styles.card}>
          {LANGUAGES.map((lang, i) => (
            <TouchableOpacity
              key={lang.code}
              style={[
                styles.optionRow,
                i < LANGUAGES.length - 1 && styles.optionBorder,
              ]}
              onPress={() => setLanguage(lang.code)}
            >
              <View>
                <Text style={styles.optionLabel}>{lang.label}</Text>
                <Text style={styles.optionSub}>{lang.native}</Text>
              </View>
              {language === lang.code && (
                <Ionicons name="checkmark-circle" size={20} color={Colors.saffron} />
              )}
            </TouchableOpacity>
          ))}
        </View>

        {/* ── Default Mode ──────────────────────────────── */}
        <SectionHeader title="Default Mode" />
        <View style={styles.card}>
          {MODES.map((m, i) => (
            <TouchableOpacity
              key={m.key}
              style={[
                styles.optionRow,
                i < MODES.length - 1 && styles.optionBorder,
              ]}
              onPress={() => setDefaultMode(m.key as QueryMode)}
            >
              <View style={{ flex: 1 }}>
                <Text style={styles.optionLabel}>{m.label}</Text>
                <Text style={styles.optionSub}>{m.description}</Text>
              </View>
              {defaultMode === m.key && (
                <Ionicons name="checkmark-circle" size={20} color={Colors.modes[m.key]} />
              )}
            </TouchableOpacity>
          ))}
        </View>

        {/* ── Preferences ───────────────────────────────── */}
        <SectionHeader title="Preferences" />
        <View style={styles.card}>
          <Row>
            <View>
              <Text style={styles.optionLabel}>Show Sources</Text>
              <Text style={styles.optionSub}>Display scripture citations below answers</Text>
            </View>
            <Switch
              value={showSources}
              onValueChange={setShowSources}
              trackColor={{ false: Colors.darkBorder, true: Colors.saffron }}
              thumbColor={Colors.white}
            />
          </Row>
          <View style={styles.optionBorder} />
          <Row>
            <View>
              <Text style={styles.optionLabel}>Haptic Feedback</Text>
              <Text style={styles.optionSub}>Vibrate on send and interactions</Text>
            </View>
            <Switch
              value={hapticFeedback}
              onValueChange={setHapticFeedback}
              trackColor={{ false: Colors.darkBorder, true: Colors.saffron }}
              thumbColor={Colors.white}
            />
          </Row>
        </View>

        {/* ── Server Status ─────────────────────────────── */}
        <SectionHeader title="Server Status" />
        <View style={styles.card}>
          {healthLoading ? (
            <ActivityIndicator color={Colors.saffron} style={{ padding: Spacing.md }} />
          ) : health ? (
            <>
              <Row>
                <Text style={styles.optionLabel}>Status</Text>
                <Text style={[styles.statusDot, { color: health.status === 'ok' ? Colors.success : Colors.error }]}>
                  ● {health.status.toUpperCase()}
                </Text>
              </Row>
              <View style={styles.optionBorder} />
              <Row>
                <Text style={styles.optionLabel}>Vector DB</Text>
                <Text style={styles.optionSub}>{health.vector_name}</Text>
              </Row>
              <View style={styles.optionBorder} />
              <Row>
                <Text style={styles.optionLabel}>Anthropic</Text>
                <Ionicons
                  name={health.anthropic ? 'checkmark-circle' : 'close-circle'}
                  size={18}
                  color={health.anthropic ? Colors.success : Colors.error}
                />
              </Row>
            </>
          ) : (
            <View style={styles.optionRow}>
              <Text style={[styles.optionSub, { color: Colors.error }]}>
                Could not reach server
              </Text>
              <TouchableOpacity onPress={checkHealth}>
                <Text style={styles.retryBtn}>Retry</Text>
              </TouchableOpacity>
            </View>
          )}
        </View>

        {/* ── About ─────────────────────────────────────── */}
        <SectionHeader title="About" />
        <View style={styles.card}>
          <TouchableOpacity
            style={styles.optionRow}
            onPress={() => Linking.openURL('https://dharmagpt.com/privacy')}
          >
            <Text style={styles.optionLabel}>Privacy Policy</Text>
            <Ionicons name="open-outline" size={16} color={Colors.ash} />
          </TouchableOpacity>
          <View style={styles.optionBorder} />
          <TouchableOpacity
            style={styles.optionRow}
            onPress={() => Linking.openURL('https://dharmagpt.com/terms')}
          >
            <Text style={styles.optionLabel}>Terms of Service</Text>
            <Ionicons name="open-outline" size={16} color={Colors.ash} />
          </TouchableOpacity>
          <View style={styles.optionBorder} />
          <View style={styles.optionRow}>
            <Text style={styles.optionLabel}>Version</Text>
            <Text style={styles.optionSub}>1.0.0 (beta)</Text>
          </View>
        </View>

        <Text style={styles.footer}>
          DharmaGPT — Wisdom rooted in sacred tradition 🕉
        </Text>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: Colors.darkBg,
  },
  content: {
    paddingBottom: Spacing.xxl,
  },
  sectionHeader: {
    fontSize: Fonts.sizes.xs,
    fontWeight: '700',
    color: Colors.ash,
    textTransform: 'uppercase',
    letterSpacing: 1,
    paddingHorizontal: Spacing.lg,
    paddingTop: Spacing.lg,
    paddingBottom: Spacing.sm,
  },
  card: {
    backgroundColor: Colors.darkCard,
    marginHorizontal: Spacing.md,
    borderRadius: Radius.lg,
    borderWidth: 1,
    borderColor: Colors.darkBorder,
    overflow: 'hidden',
  },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: Spacing.md,
    paddingVertical: 14,
  },
  optionRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: Spacing.md,
    paddingVertical: 14,
  },
  optionBorder: {
    borderTopWidth: 1,
    borderTopColor: Colors.darkBorder,
    marginHorizontal: Spacing.md,
  },
  optionLabel: {
    fontSize: Fonts.sizes.md,
    color: Colors.cream,
    fontWeight: '500',
  },
  optionSub: {
    fontSize: Fonts.sizes.xs,
    color: Colors.ash,
    marginTop: 2,
  },
  statusDot: {
    fontSize: Fonts.sizes.sm,
    fontWeight: '700',
  },
  retryBtn: {
    color: Colors.saffron,
    fontSize: Fonts.sizes.sm,
    fontWeight: '600',
  },
  footer: {
    textAlign: 'center',
    color: Colors.ash,
    fontSize: Fonts.sizes.xs,
    marginTop: Spacing.xl,
    paddingHorizontal: Spacing.xl,
  },
});
