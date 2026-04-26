import React, { useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Alert,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as Clipboard from 'expo-clipboard';
import * as Haptics from 'expo-haptics';
import Markdown from 'react-native-markdown-display';
import { Message } from '@/store/chatStore';
import { Colors, Fonts, Spacing, Radius } from '@/constants/theme';
import { dharmaApi } from '@/services/api';
import SourcesPanel from './SourcesPanel';
import { useSettingsStore } from '@/store/settingsStore';

interface Props {
  message: Message;
  onFeedbackSubmit: (msgId: string, rating: 'up' | 'down') => void;
}

export default function MessageBubble({ message, onFeedbackSubmit }: Props) {
  const isUser = message.role === 'user';
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const showSources = useSettingsStore((s) => s.showSources);
  const haptic = useSettingsStore((s) => s.hapticFeedback);

  const handleCopy = async () => {
    await Clipboard.setStringAsync(message.content);
    if (haptic) Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
  };

  const handleFeedback = async (rating: 'up' | 'down') => {
    if (message.feedback) return; // already rated
    if (!message.query_id) return;
    if (haptic) Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    onFeedbackSubmit(message.id, rating);
    try {
      await dharmaApi.feedback({
        query_id: message.query_id,
        query: '', // filled by parent if needed
        answer: message.content,
        mode: message.mode ?? 'guidance',
        sources: message.sources ?? [],
        rating,
      });
    } catch {}
  };

  if (isUser) {
    return (
      <View style={styles.userRow}>
        <View style={styles.userBubble}>
          <Text style={styles.userText}>{message.content}</Text>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.assistantRow}>
      {/* Avatar */}
      <View style={styles.avatar}>
        <Text style={styles.avatarText}>🕉</Text>
      </View>

      <View style={styles.assistantBody}>
        {/* Answer */}
        <View style={styles.assistantBubble}>
          <Markdown style={markdownStyles}>{message.content}</Markdown>
        </View>

        {/* Sources toggle */}
        {showSources && message.sources && message.sources.length > 0 && (
          <TouchableOpacity
            style={styles.sourcesToggle}
            onPress={() => setSourcesOpen((v) => !v)}
          >
            <Ionicons
              name={sourcesOpen ? 'chevron-up' : 'chevron-down'}
              size={13}
              color={Colors.saffron}
            />
            <Text style={styles.sourcesToggleText}>
              {message.sources.length} source{message.sources.length > 1 ? 's' : ''}
            </Text>
          </TouchableOpacity>
        )}

        {sourcesOpen && message.sources && (
          <SourcesPanel sources={message.sources} />
        )}

        {/* Actions row */}
        <View style={styles.actions}>
          <TouchableOpacity onPress={handleCopy} style={styles.actionBtn}>
            <Ionicons name="copy-outline" size={15} color={Colors.ash} />
          </TouchableOpacity>

          <TouchableOpacity
            onPress={() => handleFeedback('up')}
            style={styles.actionBtn}
            disabled={!!message.feedback}
          >
            <Ionicons
              name={message.feedback === 'up' ? 'thumbs-up' : 'thumbs-up-outline'}
              size={15}
              color={message.feedback === 'up' ? Colors.success : Colors.ash}
            />
          </TouchableOpacity>

          <TouchableOpacity
            onPress={() => handleFeedback('down')}
            style={styles.actionBtn}
            disabled={!!message.feedback}
          >
            <Ionicons
              name={message.feedback === 'down' ? 'thumbs-down' : 'thumbs-down-outline'}
              size={15}
              color={message.feedback === 'down' ? Colors.error : Colors.ash}
            />
          </TouchableOpacity>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  userRow: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    marginBottom: Spacing.md,
    paddingHorizontal: Spacing.md,
  },
  userBubble: {
    backgroundColor: Colors.saffron,
    borderRadius: Radius.lg,
    borderBottomRightRadius: 4,
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.sm,
    maxWidth: '78%',
  },
  userText: {
    color: Colors.white,
    fontSize: Fonts.sizes.md,
    lineHeight: 22,
  },

  assistantRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: Spacing.md,
    paddingHorizontal: Spacing.md,
  },
  avatar: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: Colors.darkCard,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: Spacing.sm,
    marginTop: 2,
    borderWidth: 1,
    borderColor: Colors.saffron,
  },
  avatarText: {
    fontSize: 14,
  },
  assistantBody: {
    flex: 1,
  },
  assistantBubble: {
    backgroundColor: Colors.darkCard,
    borderRadius: Radius.lg,
    borderTopLeftRadius: 4,
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.sm,
    borderWidth: 1,
    borderColor: Colors.darkBorder,
  },

  sourcesToggle: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingTop: 6,
    paddingLeft: 4,
  },
  sourcesToggleText: {
    fontSize: Fonts.sizes.xs,
    color: Colors.saffron,
    fontWeight: '600',
  },

  actions: {
    flexDirection: 'row',
    gap: Spacing.sm,
    paddingTop: 6,
    paddingLeft: 4,
  },
  actionBtn: {
    padding: 4,
  },
});

const markdownStyles: any = {
  body: {
    color: Colors.cream,
    fontSize: Fonts.sizes.md,
    lineHeight: 24,
  },
  strong: {
    color: Colors.gold,
    fontWeight: '700',
  },
  em: {
    color: Colors.sandal,
  },
  bullet_list: {
    marginLeft: 8,
  },
  code_inline: {
    backgroundColor: Colors.darkBg,
    color: Colors.saffronLight,
    borderRadius: 4,
    paddingHorizontal: 4,
  },
  blockquote: {
    backgroundColor: Colors.darkBg,
    borderLeftColor: Colors.saffron,
    borderLeftWidth: 3,
    paddingLeft: 8,
    marginLeft: 0,
  },
};
